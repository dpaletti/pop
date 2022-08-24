from typing import Optional

import dgl

from networks.gcn import GCN
import torch.nn as nn


class RandomNetworkDistiller(nn.Module):
    # TODO: architecture
    def __init__(
        self,
        node_features: int,
        edge_features: Optional[int],
        architecture: ...,
        name: str,
    ):
        super(RandomNetworkDistiller, self).__init__()
        self.target_network = GCN(
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture.network,
            name=name + "_target_network",
        )
        # Target Network is a frozen randomly initialized network
        self.target_network = self.target_network.requires_grad_(False)

        self.prediction_network = GCN(
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture.network,
            name=name + "_prediction_network",
        )
        self.mse_loss = nn.MSELoss()

    def forward(self, observation: dgl.DGLHeteroGraph) -> float:
        target_features = self.target_network(observation)
        predicted_features = self.prediction_network(observation)
        return self.mse_loss(predicted_features, target_features)
