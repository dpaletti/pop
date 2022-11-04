from typing import Tuple

import torch.nn as nn
import torch as th
from dgl.nn.pytorch import GraphConv
from torch import Tensor

from dgl import DGLHeteroGraph


class EGATFlatten(nn.Module):
    def __init__(self):
        super(EGATFlatten, self).__init__()

    def forward(
        self, g: DGLHeteroGraph, node_embedding: Tensor, edge_embedding: Tensor
    ) -> Tuple[Tensor, Tensor]:

        # -> (nodes, node_features * heads, batch_size),
        # -> (edges, edge_features * heads, batch_size)
        return th.flatten(node_embedding, 1), th.flatten(edge_embedding, 1)


class GATFlatten(nn.Module):
    def __init__(self):
        super(GATFlatten, self).__init__()

    def forward(self, g: DGLHeteroGraph, node_embedding: Tensor):
        # -> (nodes, node_features * heads, batch_size)
        return th.flatten(node_embedding, 1)


class EGATNodeConv(nn.Module):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        bias=True,
        allow_zero_in_degree=True,
    ):
        super(EGATNodeConv, self).__init__()
        self.convolution: GraphConv = GraphConv(
            in_feats,
            out_feats,
            bias=bias,
            allow_zero_in_degree=allow_zero_in_degree,
        )

    def forward(
        self, g: DGLHeteroGraph, node_embedding: Tensor, edge_embedding: Tensor
    ) -> Tensor:
        # -> (nodes, node_features, batch_size)
        return self.convolution(g, node_embedding, edge_weight=edge_embedding)
