from typing import Union

import torch as th
from dgl.heterograph import DGLHeteroGraph
from torch import Tensor

from pop.dueling_networks.dueling_net import DuelingNet
from pop.graph_convolutional_networks.egat_gcn import EgatGCN


class EgatDuelingGCN(DuelingNet):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        action_space_size: int,
        architecture: Union[str, dict],
        name: str,
        log_dir: str,
    ):
        super(EgatDuelingGCN, self).__init__(
            action_space_size, architecture, name, log_dir=log_dir
        )

        self._embedding: EgatGCN = EgatGCN(
            node_features,
            edge_features,
            architecture,
            name + "_embedding",
            log_dir,
        )

    @property
    def embedding(self):
        return self._embedding

    def extract_features(self, g: DGLHeteroGraph) -> Tensor:
        node_embeddings = self.embedding(g, return_mean_over_heads=False)

        graph_embedding = self.compute_graph_embedding(g, node_embeddings)

        graph_embedding = th.flatten(graph_embedding, 1)

        return graph_embedding
