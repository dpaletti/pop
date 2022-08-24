from typing import Optional

import dgl

from agents.random_network_distiller import RandomNetworkDistiller
from collections import deque
import torch as th
import numpy as np
from numpy.linalg import norm
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

from networks.inverse_model import InverseModel


class EpisodicMemory(nn.Module):

    # TODO: choose architecture
    def __init__(
        self,
        node_features: int,
        edge_features: Optional[int],
        architecture: ...,
        name: str,
    ):
        super(EpisodicMemory, self).__init__()
        self.memory = deque(maxlen=architecture.pop.episodic_memory_size)
        self.n_neighbors = architecture.n_neighbors
        self.inverse_model = InverseModel(...)
        self.k_squared_distance_running_mean = self.RunningMean()
        self.random_network_distiller = RandomNetworkDistiller(
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture.distiller,
            name=name + "_distiller",
        )
        self.distiller_error_running_mean = self.RunningMean()
        self.distiller_error_running_standard_deviation = (
            self.RunningStandardDeviation()
        )

    def _episodic_reward(
        self, current_embedding: th.Tensor, denominator_constant: float = 1e-5
    ):
        # Update Memory
        self.memory.append(current_embedding)

        # Compute K nearest neighbors wrt inverse kernel from memory
        neighbors_model = NearestNeighbors(
            n_neighbors=self.n_neighbors, metric=self._inverse_kernel
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

    def forward(self, current_state: dgl.DGLHeteroGraph):
        # Inverse Model outputs the predicted action given current state and next state
        # We then take the embedding and compare it with the ones already stored (care for the initial edge case)
        # Compute r^episodic
        # ... (now alpha)
        embedding = self.inverse_model_embedding(current_state)

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
