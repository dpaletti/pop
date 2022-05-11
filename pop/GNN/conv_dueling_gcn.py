from dgl import DGLHeteroGraph
from dgl.nn.pytorch.conv import GraphConv
from torch import Tensor
import torch.nn.functional as F

from pop.GNN.dueling_gcn import DuelingGCN
import torch.nn as nn


class ConvDuelingGCN(DuelingGCN):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        action_space_size: int,
        architecture_path: str,
        name: str,
        log_dir: str = "./",
    ):
        super(ConvDuelingGCN, self).__init__(
            node_features,
            edge_features,
            action_space_size,
            architecture_path,
            name,
            log_dir,
        )

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

    def init_advantage_stream(self, action_space_size: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(
                self.architecture["hidden_output_size"],
                self.architecture["advantage_stream_size"],
            ),
            nn.ReLU(),
            nn.Linear(self.architecture["advantage_stream_size"], action_space_size),
        )

    def init_value_stream(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(
                self.architecture["hidden_output_size"],
                self.architecture["value_stream_size"],
            ),
            nn.ReLU(),
            nn.Linear(self.architecture["value_stream_size"], 1),
        )

    def extract_features(self, g: DGLHeteroGraph) -> Tensor:
        g = self.preprocess_graph(g)

        node_conv: Tensor = F.relu(
            self.conv_1(g, DuelingGCN.dict_to_tensor(dict(g.ndata)))
        )
        edge_conv: Tensor = F.relu(
            self.conv_2(
                g, node_conv, edge_weight=DuelingGCN.dict_to_tensor(dict(g.edata))
            )
        )
        node_embeddings = self.conv_3(g, edge_conv)

        graph_embedding = self.compute_graph_embedding(g, node_embeddings)

        return graph_embedding
