from dgl.heterograph import DGLHeteroGraph
from dgl.nn.pytorch import GATv2Conv
from torch import Tensor
import torch.nn as nn
import torch as th

from GNN.dueling_gcn import DuelingGCN


class GATDuelingGCN(DuelingGCN):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        action_space_size: int,
        architecture_path: str,
        name: str,
        log_dir: str = "./",
    ):
        super(GATDuelingGCN, self).__init__(
            node_features,
            edge_features,
            action_space_size,
            architecture_path,
            name,
            log_dir,
        )

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

    def extract_features(self, g: DGLHeteroGraph) -> Tensor:
        g = self.preprocess_graph(g)

        node_embeddings: Tensor = self.attention1(g, self.dict_to_tensor(dict(g.ndata)))

        node_embeddings = th.flatten(node_embeddings, 1)

        node_embeddings: Tensor = self.attention2(g, node_embeddings)

        graph_embedding: Tensor = self.compute_graph_embedding(g, node_embeddings)

        return graph_embedding
