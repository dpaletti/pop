from typing import Tuple, Union

from dgl import DGLHeteroGraph
from dgl.nn.pytorch import EGATConv, GraphConv
from torch import Tensor

from GNN.gcn import GCN
import torch as th


class EgatGCN(GCN):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        architecture_path: str,
        name: str,
        log_dir: str,
    ):
        super(EgatGCN, self).__init__(
            node_features, edge_features, architecture_path, name, log_dir
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

    def get_embedding_dimension(self):
        return self.architecture["hidden_output_size"]

    def compute_embeddings(
        self, g: DGLHeteroGraph, return_graph: bool = False
    ) -> Union[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor, DGLHeteroGraph]]:
        g = self.preprocess_graph(g)
        node_embeddings, edge_embeddings = self.attention1(
            g,
            self.dict_to_tensor(dict(g.ndata)),
            self.dict_to_tensor(dict(g.edata)),
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
        if return_graph:
            return node_embeddings, edge_embeddings, g
        return node_embeddings, edge_embeddings

    def compute_node_embedding(
        self, g: DGLHeteroGraph, node_embeddings: th.Tensor, edge_embeddings: th.Tensor
    ) -> Tensor:
        node_embeddings = self.conv(g, node_embeddings, edge_weight=edge_embeddings)
        mean_over_heads = th.mean(node_embeddings, dim=1)
        return mean_over_heads

    def forward(self, g: DGLHeteroGraph) -> Tensor:
        node_embeddings, edge_embeddings = self.compute_embeddings(g)
        return self.compute_node_embedding(g, node_embeddings, edge_embeddings)
