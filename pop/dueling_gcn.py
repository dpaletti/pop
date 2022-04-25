import torch as th
from dgl.nn.pytorch.conv import GraphConv, GATv2Conv, EGATConv
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.heterograph import DGLHeteroGraph
from torch import Tensor, FloatTensor

from pop.gcn import GCN


class DuelingGCN(GCN):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        action_space_size: int,
        architecture_path: str,
        name: str,
        log_dir: str = "./",
    ):
        super(DuelingGCN, self).__init__(
            node_features, edge_features, architecture_path, name, log_dir
        )

        # Parameters
        self.node_features = node_features
        self.edge_features = edge_features
        self.action_space_size = action_space_size

        # Network Paths
        self.advantage_stream: nn.Module = self.init_advantage_stream(action_space_size)
        self.value_stream: nn.Module

        if self.architecture["embedding"] == "conv":

            self.conv_1 = GraphConv(
                node_features,
                edge_features,
                bias=True,
                weight=True,
                allow_zero_in_degree=True,
            ).float()
            self.conv_2 = GraphConv(
                edge_features,
                self.architecture["hidden_size"],
                bias=True,
                weight=True,
                allow_zero_in_degree=True,
            )
            self.conv_3 = GraphConv(
                self.architecture["hidden_size"],
                self.architecture["hidden_output_size"],
                allow_zero_in_degree=True,
            )

        elif self.architecture["embedding"] == "attention":
            if (
                self.architecture["hidden_node_feat_size"][2]
                != self.architecture["hidden_edge_feat_size"][2]
            ):
                raise Exception(
                    "Last value for hidden node and edge feature size must be equal"
                )
            self.attention1 = EGATConv(
                node_features,
                edge_features,
                self.architecture["hidden_node_feat_size"][0],
                self.architecture["hidden_edge_feat_size"][0],
                num_heads=self.architecture["heads"][0],
                bias=True,
            )
            self.attention2 = EGATConv(
                self.architecture["hidden_node_feat_size"][0]
                * self.architecture["heads"][0],
                self.architecture["hidden_edge_feat_size"][0]
                * self.architecture["heads"][0],
                self.architecture["hidden_node_feat_size"][1],
                self.architecture["hidden_edge_feat_size"][1],
                num_heads=self.architecture["heads"][1],
                bias=True,
            )
            self.attention3 = EGATConv(
                self.architecture["hidden_node_feat_size"][1]
                * self.architecture["heads"][1],
                self.architecture["hidden_edge_feat_size"][1]
                * self.architecture["heads"][1],
                self.architecture["hidden_node_feat_size"][2],
                self.architecture["hidden_edge_feat_size"][2],
                num_heads=self.architecture["heads"][2],
                bias=True,
            )

            self.conv = GraphConv(
                self.architecture["hidden_node_feat_size"][2],
                self.architecture["hidden_output_size"],
                bias=True,
                weight=True,
                allow_zero_in_degree=True,
            )

        elif self.architecture["embedding"] == "gat":
            self.attention1 = GATv2Conv(
                self.architecture["node_features"],
                self.architecture["hidden_node_feat_size"][0],
                num_heads=self.architecture["heads"][0],
                residual=True,
                activation=nn.ReLU(),
                allow_zero_in_degree=True,
                bias=True,
                share_weights=True,
            )
            # Here we concatenate edge features with node features
            self.attention2 = GATv2Conv(
                self.architecture["hidden_node_feat_size"][1]
                * self.architecture["heads"][0],
                self.architecture["hidden_output_size"],
                num_heads=self.architecture["heads"][1],
                residual=True,
                activation=nn.ReLU(),
                allow_zero_in_degree=True,
                bias=True,
                share_weights=True,
            )

        self.value_stream = nn.Sequential(
            nn.Linear(
                self.architecture["hidden_output_size"]
                * (
                    self.architecture["heads"][-1]
                    if self.architecture["embedding"] != "conv"
                    else 1
                ),
                self.architecture["value_stream_size"],
            ),
            nn.ReLU(),
            nn.Linear(self.architecture["value_stream_size"], 1),
        )

    def init_advantage_stream(self, action_space_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(
                self.architecture["hidden_output_size"]
                * (
                    self.architecture["heads"][-1]
                    if self.architecture["embedding"] != "conv"
                    else 1
                ),
                self.architecture["advantage_stream_size"],
            ),
            nn.ReLU(),
            nn.Linear(self.architecture["advantage_stream_size"], action_space_size),
        )

    def preprocess_graph(self, g: DGLHeteroGraph) -> DGLHeteroGraph:
        num_nodes = g.batch_num_nodes()
        num_edges = g.batch_num_edges()
        g = dgl.add_self_loop(g)
        g.set_batch_num_nodes(num_nodes)
        g.set_batch_num_edges(num_edges)
        return g

    def extract_features(self, g: DGLHeteroGraph) -> Tensor:
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
        if self.architecture["embedding"] == "conv":
            node_conv: Tensor = F.relu(
                self.conv_1(g, DuelingGCN.dict_to_tensor(dict(g.ndata)))
            )
            edge_conv: Tensor = F.relu(
                self.conv_2(
                    g, node_conv, edge_weight=DuelingGCN.dict_to_tensor(dict(g.edata))
                )
            )
            node_embeddings = self.conv_3(g, edge_conv)

        elif self.architecture["embedding"] == "attention":
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
        elif self.architecture["embedding"] == "gat":
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

        if (
            self.architecture["embedding"] == "attention"
            or self.architecture["embedding"] == "gat"
        ):
            graph_embedding = th.flatten(graph_embedding, 1)

        return graph_embedding

    def forward(self, g: DGLHeteroGraph) -> Tensor:
        graph_embedding: Tensor = self.extract_features(g)

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

        graph_embedding: Tensor = self.extract_features(g)

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
        }
        checkpoint = checkpoint | self.architecture
        th.save(checkpoint, self.log_file)

    def load(
        self,
        log_dir: str,
    ):
        checkpoint = th.load(log_dir)
        self.name = checkpoint["name"]
        self.load_state_dict(checkpoint["network_state"])
        self.node_features = checkpoint["node_features"]
        self.edge_features = checkpoint["edge_features"]
        self.action_space_size = checkpoint["action_space_size"]
        self.architecture = {
            i: j
            for i, j in checkpoint.items()
            if i
            not in {
                "network_state",
                "node_features",
                "edge_features",
                "action_space_size",
                "name",
            }
        }
        print("Network Succesfully Loaded!")
