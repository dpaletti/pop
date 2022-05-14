from typing import Union, Tuple
import torch.nn as nn
import json

from dgl import DGLHeteroGraph
from torch import Tensor
from graph_convolutional_networks.egat_gcn import EgatGCN

# TODO: for pointer nets https://ychai.uk/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/
# Alternating Learning: chi gioca ha un learning rate più alto
# Schedule a Turni
# Fissare le comunità
from managers.manager import Manager
from managers.node_attention import NodeAttention


class CommunityManager(Manager):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        architecture: Union[str, dict],
        name: str,
        log_dir: str,
    ):
        super(CommunityManager, self).__init__(
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture,
            name=name,
            log_dir=log_dir,
        )
        self.architecture = (
            json.load(open(architecture)) if type(architecture) is str else architecture
        )
        self._embedding = EgatGCN(
            node_features,
            edge_features,
            self.architecture["embedding_architecture"],
            name,
            log_dir,
        ).float()

        self._node_attention = NodeAttention(
            architecture, self.embedding.get_embedding_dimension()
        )

    def get_embedding_dimension(self):
        return self.embedding.get_embedding_dimension()

    @property
    def embedding(self):
        return self._embedding

    @property
    def node_attention(self):
        return self._node_attention

    def forward(self, g: DGLHeteroGraph) -> Tuple[int, DGLHeteroGraph]:

        # -> (Nodes, Embedding Size, (optional) Batch Size)
        node_embedding: Tensor = self.embedding(g)

        best_node: int = self.node_attention(node_embedding)

        g.ndata["embedding"] = node_embedding

        return g.nodes[best_node].data["action"], g
