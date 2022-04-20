import torch as th
from dgl.nn.pytorch.conv import GraphConv, GATv2Conv, EGATConv
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.heterograph import DGLHeteroGraph
from torch import Tensor, FloatTensor
from typing import List, Optional
from pathlib import Path
from prettytable import PrettyTable


class DuelingGCN(nn.Module):
    """
    Dueling Graph Convolutional Neural Network Implementation
    For L2RPN ICAPS Environment:
    1. node_features = 4
    2. edge_features = 11
    3. action_space_size = "id_converter.n" (to find out when using id_converter)

    Attributes
    ----------
    conv1: :class:`GraphConv`
        graph convolution over node features
    conv2: :class:`GraphConv`
        graph convolution over edge features
    conv3: :class:`GraphConv`
        graph convolution over topology
    value_stream: :class:`nn.Sequential`
        Non-linearity + Linearity to compute state-value
    advantage_stream: :class:`nn.Sequential`
        Non-linearity + Linearity to compute advantage value for each state-action pair
    """

    def __init__(
        self,
        node_features: int,
        edge_features: int,
        action_space_size: Optional[int] = None,
        name: str = "dueling_gcn",
        embedding: str = "conv",
        pooling: str = "avg",
        hidden_size: int = 512,
        hidden_output_size: int = 1024,
        hidden_node_feat_size: List[int] = [64, 128, 256],
        hidden_edge_feat_size: List[int] = [64, 128, 256],
        heads: List[int] = [3, 6, 12],
        value_stream_size: int = 2048,
        advantage_stream_size: int = 2048,
        log_dir: str = "dueling_gcn.pt",
    ):
        super(DuelingGCN, self).__init__()

        # Logging path
        self.log_dir: str = log_dir
        self.name: str = name
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.log_dir = str(Path(self.log_dir, name + ".pt"))

        # Hyperparameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.action_space_size = action_space_size
        self.embedding = embedding
        self.pooling = pooling

        ## Conv Hyperparameters
        self.hidden_size = hidden_size

        ## Conv and Attention Hyperparameters
        self.hidden_output_size = hidden_output_size

        ## Attention Hyperparameters
        self.hidden_node_feat_size = hidden_node_feat_size
        self.hidden_edge_feat_size = hidden_edge_feat_size

        ## Value and Advantage Hyperparameters
        self.value_stream_size = value_stream_size
        self.advantage_stream_size = advantage_stream_size

        if embedding == "conv":
            # WARNING allow_zero_in_degree allows invalid outputs coming from 0 in-degree nodes

            self.conv_1 = GraphConv(
                node_features,
                edge_features,
                bias=True,
                weight=True,
                allow_zero_in_degree=True,
            ).float()
            self.conv_2 = GraphConv(
                edge_features,
                hidden_size,
                bias=True,
                weight=True,
                allow_zero_in_degree=True,
            )
            self.conv_3 = GraphConv(
                hidden_size,
                hidden_output_size,
                allow_zero_in_degree=True,
            )

        elif embedding == "attention":
            if hidden_node_feat_size[2] != hidden_edge_feat_size[2]:
                raise Exception(
                    "Last value for hidden node and edge feature size must be equal"
                )
            self.attention1 = EGATConv(
                node_features,
                edge_features,
                hidden_node_feat_size[0],
                hidden_edge_feat_size[0],
                num_heads=heads[0],
                bias=True,
            )
            self.attention2 = EGATConv(
                hidden_node_feat_size[0] * heads[0],
                hidden_edge_feat_size[0] * heads[0],
                hidden_node_feat_size[1],
                hidden_edge_feat_size[1],
                num_heads=heads[1],
                bias=True,
            )
            self.attention3 = EGATConv(
                hidden_node_feat_size[1] * heads[1],
                hidden_edge_feat_size[1] * heads[1],
                hidden_node_feat_size[2],
                hidden_edge_feat_size[2],
                num_heads=heads[2],
                bias=True,
            )
            self.last_head = heads[2]

            self.conv = GraphConv(
                hidden_node_feat_size[2],
                hidden_output_size,
                bias=True,
                weight=True,
                allow_zero_in_degree=True,
            )

        elif embedding == "gat":
            self.attention1 = GATv2Conv(
                node_features,
                hidden_node_feat_size[0],
                num_heads=heads[0],
                residual=True,
                activation=nn.ReLU(),
                allow_zero_in_degree=True,
                bias=True,
                share_weights=True,
            )
            # Here we concatenate edge features with node features
            self.attention2 = GATv2Conv(
                hidden_node_feat_size[0] * heads[0],
                hidden_output_size,
                num_heads=heads[1],
                residual=True,
                activation=nn.ReLU(),
                allow_zero_in_degree=True,
                bias=True,
                share_weights=True,
            )
            self.last_head = heads[1]

        self.value_stream = nn.Sequential(
            nn.Linear(
                self.hidden_output_size
                * (self.last_head if self.embedding != "conv" else 1),
                value_stream_size,
            ),
            nn.ReLU(),
            nn.Linear(value_stream_size, 1),
        )

        if action_space_size is not None:
            self.init_advantage_stream(action_space_size)
        else:
            print(
                "CARE: you need to call init_advantage_stream in order to have a functioning network"
            )

    def init_advantage_stream(self, action_space_size: int):
        self.advantage_stream = nn.Sequential(
            nn.Linear(
                self.hidden_output_size
                * (self.last_head if self.embedding != "conv" else 1),
                self.advantage_stream_size,
            ),
            nn.ReLU(),
            nn.Linear(self.advantage_stream_size, action_space_size),
        )
        self.action_space_size = action_space_size

    def _extract_features(self, g: DGLHeteroGraph) -> Tensor:
        """
        Embed graph by graph convolutions

        Parameters
        ----------
        g: :class:`DGLHeteroGraph`
            input graph with node and edge features

        Return
        ------
        graph_embedding: :class:`Tensor`
            graph embedding computed from node embeddings
        """
        node_embeddings: Tensor
        num_nodes = g.batch_num_nodes()
        num_edges = g.batch_num_edges()
        g = dgl.add_self_loop(g)
        g.set_batch_num_nodes(num_nodes)
        g.set_batch_num_edges(num_edges)
        if self.embedding == "conv":
            node_conv: Tensor = F.relu(
                self.conv_1(g, DuelingGCN.dict_to_tensor(dict(g.ndata)))
            )
            edge_conv: Tensor = F.relu(
                self.conv_2(
                    g, node_conv, edge_weight=DuelingGCN.dict_to_tensor(dict(g.edata))
                )
            )
            node_embeddings = self.conv_3(g, edge_conv)

        elif self.embedding == "attention":
            edge_embeddings: Tensor

            node_embeddings, edge_embeddings = self.attention1(
                g,
                DuelingGCN.dict_to_tensor(dict(g.ndata)),
                DuelingGCN.dict_to_tensor(dict(g.edata)),
            )

            node_embeddings = th.flatten(node_embeddings, 1)
            edge_embeddings = th.flatten(edge_embeddings, 1)

            node_embeddings, edge_embeddings = self.attention2(
                g, node_embeddings, edge_embeddings
            )

            node_embeddings = th.flatten(node_embeddings, 1)
            edge_embeddings = th.flatten(edge_embeddings, 1)

            node_embeddings, edge_embeddings = self.attention3(
                g, node_embeddings, edge_embeddings
            )

            node_embeddings = self.conv(g, node_embeddings, edge_weight=edge_embeddings)
        elif self.embedding == "gat":
            node_embeddings: Tensor = self.attention1(
                g, DuelingGCN.dict_to_tensor(dict(g.ndata))
            )

            node_embeddings = th.flatten(node_embeddings, 1)

            node_embeddings = th.flatten(node_embeddings, 1)

            node_embeddings: Tensor = self.attention2(g, node_embeddings)

        else:
            raise Exception("Embedding is not among the available ones")

        g.ndata["node_embeddings"] = node_embeddings
        graph_embedding: Tensor = dgl.mean_nodes(g, "node_embeddings")
        del g.ndata["node_embeddings"]

        if self.embedding == "attention" or self.embedding == "gat":
            graph_embedding = th.flatten(graph_embedding, 1)

        return graph_embedding

    def forward(self, g: DGLHeteroGraph) -> Tensor:
        graph_embedding: Tensor = self._extract_features(g)

        # Compute value of current state
        state_value: float = self.value_stream(graph_embedding)

        # Compute advantage of (current_state, action) for each action
        state_advantages: FloatTensor = self.advantage_stream(graph_embedding)

        q_values: Tensor = state_value + (state_advantages - state_advantages.mean())
        return q_values

    def advantage(self, g: DGLHeteroGraph) -> Tensor:
        """
        Advantage value for each state-action pair

        Parameters
        ----------
        g: :class:`DGLHeteroGraph`
            input graph

        Return
        ------
        state_advantages: :class: `Tensor`
            state-action advantages

        """

        graph_embedding: Tensor = self._extract_features(g)

        # Compute advantage of (current_state, action) for each action
        state_advantages: FloatTensor = self.advantage_stream(graph_embedding)

        return state_advantages

    def save(self):
        """
        Save a checkpoint of the network.
        Save all the hyperparameters.
        """
        checkpoint = {
            "name": self.name,
            "network_state": self.state_dict(),
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "action_space_size": self.action_space_size,
            "embedding": self.embedding,
            "pooling": self.pooling,
            "hidden_size": self.hidden_size,
            "hidden_output_size": self.hidden_output_size,
            "hidden_node_feat_size": self.hidden_node_feat_size,
            "hidden_edge_feat_size": self.hidden_edge_feat_size,
            "value_stream_size": self.value_stream_size,
            "advantage_stream_size": self.advantage_stream_size,
        }
        th.save(checkpoint, self.log_dir)

    def load(
        self,
        log_dir: str,
    ):
        checkpoint = th.load(log_dir)
        self.load_state_dict(checkpoint["network_state"])
        self.node_features = checkpoint["node_features"]
        self.edge_features = checkpoint["edge_features"]
        self.action_space_size = checkpoint["action_space_size"]
        self.embedding = checkpoint["embedding"]
        self.pooling = checkpoint["pooling"]
        self.hidden_size = checkpoint["hidden_size"]
        self.hidden_output_size = checkpoint["hidden_output_size"]
        self.hidden_node_feat_size = checkpoint["hidden_node_feat_size"]
        self.hidden_edge_feat_size = checkpoint["hidden_edge_feat_size"]
        self.value_stream_size = checkpoint["value_stream_size"]
        self.advantage_stream_size = checkpoint["advantage_stream_size"]
        print("Network Succesfully Loaded!")

    @staticmethod
    def dict_to_tensor(d: dict) -> th.Tensor:
        """
        Convert node/edge features represented as a dict from Deep Graph Library
        to a tensor

        Parameters
        ----------
        d: ``dict``
            input dictionary with keys as feature names and values as feature values
        """

        return (
            th.stack([column.data for column in list(d.values())])
            .transpose(0, 1)
            .float()
        )

    @staticmethod
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        non_trainable_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                params = parameter.numel()
                table.add_row([name + " (non trainable)", params])
                non_trainable_params += params
            else:
                params = parameter.numel()
                table.add_row([name, params])
                total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        print(f"Total Non Trainable Params: {non_trainable_params}")
        return total_params, non_trainable_params
