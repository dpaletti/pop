from typing import Optional, Tuple

import dgl
from torch import Tensor

from agents.random_network_distiller import RandomNetworkDistiller
from collections import deque
import torch as th
import numpy as np
from numpy.linalg import norm
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

from configs.agent_architecture import EpisodicMemoryArchitecture
from networks.gcn import GCN
from networks.network_architecture_parsing import get_network


# TODO: Make this an ExplorationModule
# TODO: Refactor BaseAgent to have ExplorationModules instead of normal Exploration
class EpisodicMemory(nn.Module):

    # TODO: choose architecture
    def __init__(
        self,
        node_features: int,
        edge_features: Optional[int],
        architecture: EpisodicMemoryArchitecture,
        name: str,
    ):
        super(EpisodicMemory, self).__init__()

        self.memory = deque(maxlen=architecture.size)
        self.neighbors = architecture.neighbors
        self.exploration_bonus_limit = architecture.exploration_bonus_limit

        self.inverse_model = self.InverseNetwork(
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture.inverse_model,
            name=name + "_inverse_model",
            log_dir=None,
        )
        self.k_squared_distance_running_mean = self.RunningMean()

        self.random_network_distiller = RandomNetworkDistiller(
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture.random_network_distiller,
            name=name + "_distiller",
        )
        self.distiller_error_running_mean = self.RunningMean()
        self.distiller_error_running_standard_deviation = (
            self.RunningStandardDeviation()
        )

    def forward(
        self,
        current_state: dgl.DGLHeteroGraph,
        next_state: dgl.DGLHeteroGraph,
        action: int,
    ):
        predicted_action, current_state_embedding = self.inverse_model(
            current_state, next_state
        )
        self.inverse_model.learn(action, predicted_action)
        episodic_reward = self._episodic_reward(current_state_embedding)
        exploration_bonus = self._exploration_bonus(current_state)
        return episodic_reward * th.clip(
            exploration_bonus, 1, self.exploration_bonus_limit
        )

    def _episodic_reward(
        self, current_embedding: th.Tensor, denominator_constant: float = 1e-5
    ):
        # Update Memory
        self.memory.append(current_embedding)

        # Compute K nearest neighbors wrt inverse kernel from memory
        neighbors_model = NearestNeighbors(
            n_neighbors=self.neighbors, metric=self._inverse_kernel
        )
        memory_array = np.array(self.memory)
        neighbors_model.fit(memory_array)
        neighbor_distances, _ = memory_array[
            neighbors_model.kneighbors(
                np.array(current_embedding).reshape(1, -1),
            )
        ].flatten()

        # Episodic Reward
        return 1 / np.sqrt(np.sum(neighbor_distances) + denominator_constant)

    def _exploration_bonus(self, current_state: dgl.DGLHeteroGraph):
        distiller_error = self.random_network_distiller(current_state)
        self.distiller_error_running_mean.update(distiller_error)
        self.distiller_error_running_standard_deviation.update(distiller_error)
        return (
            1
            + (distiller_error - self.distiller_error_running_mean)
            / self.distiller_error_running_standard_deviation
        )

    def _inverse_kernel(self, x: np.array, y: np.array, epsilon: float = 1e-5) -> float:
        # x <- current_state_embedding as computed by the inverse model
        # y <- K nearest neighbors to x in M (all Ks, then the outputs of the kernels are summed)
        squared_euclidean_distance = norm(x - y) ** 2
        self.k_squared_distance_running_mean.update(squared_euclidean_distance)
        return epsilon / (
            (squared_euclidean_distance / self.k_squared_distance_running_mean.value)
            + epsilon
        )

    class InverseNetwork(nn.Module):
        def __init__(
            self,
            node_features: int,
            edge_features: Optional[int],
            architecture: ...,
            name: str,
            log_dir: Optional[str] = None,
        ):
            super(self.__class__, self).__init__()
            self.embedding_network = GCN(
                node_features=node_features,
                edge_features=edge_features,
                architecture=architecture.embedding,
                name=name + "_current_state_embedding",
                log_dir=log_dir,
            )

            self.action_prediction_stream: nn.Sequential = get_network(
                self,
                architecture=architecture.action_prediction_stream,
                is_graph_network=False,
            )
            self.loss = nn.CrossEntropyLoss()

        def forward(
            self, current_state: dgl.DGLHeteroGraph, next_state: dgl.DGLHeteroGraph
        ) -> Tuple[Tensor, Tensor]:
            current_state_embedding: th.Tensor = self.embedding_network(current_state)
            next_state_embedding: th.Tensor = self.embedding_network(next_state)
            predicted_action = self.action_prediction_stream(
                th.stack((current_state_embedding, next_state_embedding))
            )
            return predicted_action, current_state_embedding

        def learn(self, action: int, predicted_action: th.Tensor):
            self.loss(action, predicted_action)
            self.loss.backward()

    class RunningMean:
        def __init__(self):
            self.running_sum: float = 0
            self.running_count: int = 0

        @property
        def value(self):
            return self.running_sum / self.running_count

        def update(self, value_to_add: float):
            self.running_sum += value_to_add
            self.running_count += 1

    class RunningStandardDeviation:
        def __init__(self):
            self.running_sum: float = 0
            self.running_sum_of_squares: float = 0
            self.running_count: int = 0

        @property
        def value(self):
            return np.sqrt(
                (self.running_sum_of_squares / self.running_count)
                - (self.running_sum / self.running_count) ** 2
            )

        def update(self, value_to_add: float):
            self.running_sum += value_to_add
            self.running_sum_of_squares += value_to_add**2
            self.running_count += 1
