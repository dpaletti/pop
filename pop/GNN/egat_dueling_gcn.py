import torch as th
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch.conv import EGATConv
import torch.nn as nn
from dgl.heterograph import DGLHeteroGraph
from torch import Tensor

from GNN.dueling_gcn import DuelingGCN


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

    def init_value_stream(self) -> nn.Module:
        return nn.Sequential(
            nn.Linear(
                self.architecture["hidden_output_size"]
                * self.architecture["heads"][-1],
                self.architecture["value_stream_size"],
            ),
            nn.ReLU(),
            nn.Linear(self.architecture["value_stream_size"], 1),
        )

    def extract_features(self, g: DGLHeteroGraph) -> Tensor:
        g = self.preprocess_graph(g)
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

        graph_embedding = self.compute_graph_embedding(g, node_embeddings)

        graph_embedding = th.flatten(graph_embedding, 1)

        return graph_embedding
