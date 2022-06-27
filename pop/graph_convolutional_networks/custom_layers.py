from typing import Tuple

import torch.nn as nn
import torch as th
from dgl.nn.pytorch import GraphConv
from torch import Tensor

from typings.dgl.heterograph import DGLHeteroGraph


class EGATFlatten(nn.Module):
    def __init__(self):
        super(EGATFlatten, self).__init__()

    def forward(
        self, node_edge_embedding_pair: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        # -> (nodes, heads, node_features, batch_size),
        # -> (edges, heads, edge_features, batch_size)
        node_embedding, edge_embedding = node_edge_embedding_pair

        # -> (nodes, node_features * heads, batch_size),
        # -> (edges, edge_features * heads, batch_size)
        return th.flatten(node_embedding, 1), th.flatten(edge_embedding, 1)


class EGATNodeConv(nn.Module):
    def __init__(
        self,
        in_feats: int,
        out_feats: int,
        bias=True,
        weight=True,
        allow_zero_in_degree=True,
    ):
        super(EGATNodeConv, self).__init__()
        self.convolution: GraphConv = GraphConv(
            in_feats,
            out_feats,
            bias=bias,
            weight=weight,
            allow_zero_in_degree=allow_zero_in_degree,
        )

    def forward(
        self, g: DGLHeteroGraph, node_edge_embedding_pair: Tuple[Tensor, Tensor]
    ):
        # -> (nodes, node_features, batch_size),
        # -> (edges, edge_features, batch_size)
        node_embedding, edge_embedding = node_edge_embedding_pair

        # -> (nodes, node_features, batch_size)
        return self.convolution(g, node_embedding, edge_weight=edge_embedding)
