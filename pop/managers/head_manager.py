from typing import Union, Optional, Tuple, Any

from dgl import DGLHeteroGraph
from torch import Tensor

from pop.networks.gat_gcn import GatGCN
import json

from pop.managers.manager import Manager
from pop.managers.node_attention import NodeAttention

import torch as th
from torchinfo import summary


class HeadManager(Manager):
    def __init__(
        self,
        node_features: int,
        architecture: Union[str, dict],
        name: str,
        log_dir: str,
        training: bool,
    ):
        super(HeadManager, self).__init__(
            node_features=node_features,
            edge_features=None,
            architecture=architecture,
            name=name,
            log_dir=log_dir,
            training=training,
        )

        self.architecture = (
            json.load(open(architecture)) if type(architecture) is str else architecture
        )

        self._embedding = GatGCN(
            node_features,
            architecture["embedding_architecture"],
            name + "_embedding",
            log_dir,
        )

        self._node_attention = NodeAttention(
            architecture, self.embedding.get_embedding_dimension(), training=training
        )

    @property
    def embedding(self):
        return self._embedding

    @property
    def node_choice(self):
        return self._node_attention

    def get_summary(self):
        return summary(self)

    def get_extra_state(self) -> Any:
        return None

    def forward(self, g: DGLHeteroGraph) -> Tuple[int, int]:
        node_embeddings: Tensor = self.embedding(g, return_mean_over_heads=True)
        self.current_best_node: int = self.node_choice(node_embeddings)
        best_node = int(self.current_best_node)

        return int(g.nodes[best_node].data["action"].squeeze()[-1].item()), best_node
