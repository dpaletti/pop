import copy
from typing import Optional

import dgl
import torch as th

from configs.agent_architecture import RandomNetworkDistillerArchitecture
from networks.gcn import GCN
import torch.nn as nn


class RandomNetworkDistiller(nn.Module):
    def __init__(
        self,
        node_features: int,
        edge_features: Optional[int],
        architecture: RandomNetworkDistillerArchitecture,
        name: str,
    ):
        super(RandomNetworkDistiller, self).__init__()

        self.prediction_network = GCN(
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture.network,
            name=name + "_distiller_prediction",
        )

        # Target Network is a frozen randomly initialized network
        self.target_network = GCN(
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture.network,
            name=name + "distiller_target",
        )
        self.target_network = self.target_network.requires_grad_(False)

        self.distiller_optimizer: th.optim.Optimizer = th.optim.Adam(
            self.prediction_network.parameters(),
            lr=architecture.learning_rate,
        )

        self.mse_loss = nn.MSELoss()
        self.last_loss: Optional[th.Tensor] = None

    def forward(self, observation: dgl.DGLHeteroGraph) -> float:
        target_features = self.target_network(observation)
        predicted_features = self.prediction_network(observation)
        self.last_loss = self.mse_loss(target_features, predicted_features)
        return self.last_loss

    def learn(self):
        self.distiller_optimizer.zero_grad()
        self.last_loss.backward()
        self.distiller_optimizer.step()
