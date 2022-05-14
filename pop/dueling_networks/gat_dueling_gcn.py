from typing import Union

from dgl.heterograph import DGLHeteroGraph
from torch import Tensor

from dueling_networks.dueling_net import DuelingNet
from graph_convolutional_networks.gat_gcn import GatGCN


class GatDuelingGCN(DuelingNet):
    def __init__(
        self,
        node_features: int,
        action_space_size: int,
        architecture: Union[str, dict],
        name: str,
        log_dir: str = "./",
    ):
        super(GatDuelingGCN, self).__init__(
            action_space_size,
            architecture,
            name,
            log_dir,
        )

        self._embedding = GatGCN(
            node_features, architecture, name + "_embedding", log_dir
        )

    @property
    def embedding(self):
        return self._embedding

    def extract_features(self, g: DGLHeteroGraph) -> Tensor:
        node_embeddings: Tensor = self.embedding(g)

        graph_embedding: Tensor = self.compute_graph_embedding(g, node_embeddings)

        return graph_embedding
