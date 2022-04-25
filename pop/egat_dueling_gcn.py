import torch as th
from dgl.nn.pytorch.conv import EGATConv
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.heterograph import DGLHeteroGraph
from torch import Tensor, FloatTensor

from dueling_gcn import DuelingGCN


class EgatDuelingGCN(DuelingGCN):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        action_space_size: int,
        architecture_path: str,
        name: str,
        log_dir: str = "./",
    ):
        super(EgatDuelingGCN, self).__init__(
            node_features,
            edge_features,
            action_space_size,
            architecture_path,
            name,
            log_dir,
        )

        self.advantage_stream: nn.Module = self.init_advantage_stream(action_space_size)
        self.value_stream: nn.Module = self.init_value_stream()

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
