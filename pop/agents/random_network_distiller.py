import copy
from typing import Optional

import dgl

from configs.network_architecture import NetworkArchitecture
from networks.gcn import GCN
import torch.nn as nn


class RandomNetworkDistiller(nn.Module):
    # TODO: architecture
    def __init__(
        self,
        node_features: int,
        edge_features: Optional[int],
        architecture: NetworkArchitecture,
        name: str,
    ):
        super(RandomNetworkDistiller, self).__init__()
        self.prediction_network = GCN(
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture,
            name=name + "_distiller",
        )

        # Target Network is a frozen randomly initialized network
        self.target_network = copy.deepcopy(self.prediction_network)
        self.target_network = self.target_network.requires_grad_(False)

        self.mse_loss = nn.MSELoss()

    def forward(self, observation: dgl.DGLHeteroGraph) -> float:
        target_features = self.target_network(observation)
        predicted_features = self.prediction_network(observation)
        loss = self.mse_loss(target_features, predicted_features)
        loss.backward()
        return loss
