from typing import Optional

import torch.nn as nn

from networks.gcn import GCN
from networks.network_architecture_parsing import get_network


class InverseModel(nn.Module):
    def __init__(
        self,
        node_features: int,
        edge_features: Optional[int],
        architecture: ...,
        name: str,
        log_dir: Optional[str] = None,
    ):
        super(InverseModel, self).__init__()
        self.current_state_embedding = GCN(
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture.embedding,
            name=name + "_current_state_embedding",
            log_dir=log_dir,
        )

        self.next_state_embedding = GCN(
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture.embedding,
            name=name + "_next_state_embedding",
            log_dir=log_dir,
        )

        self.action_prediction_stream: nn.Sequential = get_network(
            self,
            architecture=architecture.action_prediction_stream,
            is_graph_network=False,
        )
    def forward(self, current_state):

