from typing import Union

from torch import Tensor

from pop.graph_convolutional_networks.gcn import GCN
from dgl.nn.pytorch import GATv2Conv
import torch.nn as nn
import torch as th
import dgl


class GatGCN(GCN):
    def __init__(
        self,
        node_features: int,
        architecture: Union[str, dict],
        name: str,
        log_dir: str,
        **kwargs  # Compliance with GCN signature
    ):
        super(GatGCN, self).__init__(
            node_features=node_features,
            edge_features=None,
            architecture=architecture,
            name=name,
            log_dir=log_dir,
        )

        self.attention1 = GATv2Conv(
            node_features,
            self.architecture["hidden_node_feat_size"][0],
            num_heads=self.architecture["heads"][0],
            residual=True,
            activation=nn.ReLU(),
            allow_zero_in_degree=True,
            bias=True,
            share_weights=True,
        )

        self.attention2 = GATv2Conv(
            self.architecture["hidden_node_feat_size"][0]
            * self.architecture["heads"][0],
            self.architecture["hidden_output_size"],
            num_heads=self.architecture["heads"][1],
            residual=True,
            activation=nn.ReLU(),
            allow_zero_in_degree=True,
            bias=True,
            share_weights=True,
        )

    def forward(self, g: dgl.DGLHeteroGraph, return_mean_over_heads=False) -> Tensor:
        self.add_self_loop_to_batched_graph(g)

        node_embeddings: Tensor = self.attention1(
            g,
            th.flatten(self.to_tensor(dict(g.ndata)), start_dim=1),
        )

        node_embeddings = th.flatten(node_embeddings, 1)

        node_embeddings: Tensor = self.attention2(g, node_embeddings)

        if return_mean_over_heads:
            return th.mean(node_embeddings, dim=1)
        return node_embeddings

    def get_embedding_dimension(self):
        return self.architecture["hidden_output_size"]
