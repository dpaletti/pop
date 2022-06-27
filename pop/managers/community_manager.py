from typing import Union, Tuple, Optional
import json

from dgl import DGLHeteroGraph
from torch import Tensor
import torch as th
from pop.graph_convolutional_networks.egat_gcn import EgatGCN
from pop.managers.manager import Manager
from pop.managers.node_attention import NodeAttention


# TODO: for pointer nets https://ychai.uk/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/
# Alternating Learning: actor playing has a higher learning rate
# Turn-Based Schedule

# TODO: Q learning for managers with masked output
class CommunityManager(Manager):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        architecture: Union[str, dict],
        name: str,
        log_dir: Optional[str],
        training: bool,
        **kwargs
    ):
        super(CommunityManager, self).__init__(
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture,
            name=name,
            log_dir=log_dir,
            training=training,
        )

        self.architecture = (
            json.load(open(architecture)) if type(architecture) is str else architecture
        )
        self._embedding = EgatGCN(
            node_features,
            edge_features,
            self.architecture["embedding_architecture"],
            name + "_embedding",
            log_dir,
        ).float()

        self._node_attention = NodeAttention(
            architecture, self.embedding.get_embedding_dimension(), training=training
        )

    def get_embedding_dimension(self):
        return self.embedding.get_embedding_dimension()

    @property
    def embedding(self):
        return self._embedding

    @property
    def node_choice(self) -> NodeAttention:
        return self._node_attention

    def forward(self, g: DGLHeteroGraph) -> Tuple[int, DGLHeteroGraph, int]:
        node_embedding: Tensor
        edge_attention: Tensor

        # -> (Nodes, Embedding Size, (optional) Batch Size), (Edges, Heads, 1, (optional) Batch Size)
        node_embedding, edge_attention = self.embedding(g, get_attention=True)

        self.current_best_node: int = self.node_choice(node_embedding)

        g.ndata["embedding"] = node_embedding.detach()
        best_action: int = g.nodes[self.current_best_node].data["action"].item()

        return best_action, g, self.current_best_node
