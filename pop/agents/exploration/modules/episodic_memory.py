from typing import Optional, Tuple, Callable, Any

import dgl
from torch import Tensor

from agents.base_gcn_agent import BaseGCNAgent
from agents.exploration.exploration_module import ExplorationModule
from agents.exploration.random_network_distiller import RandomNetworkDistiller
from collections import deque
import torch as th
import numpy as np
from numpy.linalg import norm
import torch.nn as nn
from sklearn.neighbors import NearestNeighbors

from configs.agent_architecture import (
    EpisodicMemoryParameters,
    InverseModelArchitecture,
)
from networks.gcn import GCN
from networks.network_architecture_parsing import get_network
from typing import cast


class EpisodicMemory(nn.Module, ExplorationModule):
    def __init__(
        self,
        agent: BaseGCNAgent,
    ):
        super(EpisodicMemory, self).__init__()

        exploration_parameters: EpisodicMemoryParameters = cast(
            EpisodicMemoryParameters, agent.architecture.exploration
        )
        node_features = agent.node_features
        edge_features = agent.edge_features
        name = agent.name + "_episodic_memory"

        self.memory = deque(maxlen=exploration_parameters.size)
        self.neighbors = exploration_parameters.neighbors
        self.exploration_bonus_limit = exploration_parameters.exploration_bonus_limit

        self.inverse_model = self.InverseNetwork(
            node_features=node_features,
            edge_features=edge_features,
            actions=agent.actions,
            architecture=exploration_parameters.inverse_model,
            name=name + "_inverse_model",
            log_dir=None,
        )
        self.k_squared_distance_running_mean = self.RunningMean()

        self.random_network_distiller = RandomNetworkDistiller(
            node_features=node_features,
            edge_features=edge_features,
            architecture=exploration_parameters.random_network_distiller,
            name=name + "_distiller",
        )
        self.distiller_error_running_mean = self.RunningMean()
        self.distiller_error_running_standard_deviation = (
            self.RunningStandardDeviation()
        )

        self.last_predicted_action: Optional[th.Tensor] = None

    def update(self, action: int) -> None:

        self.random_network_distiller.learn()
        self.inverse_model.learn(action, self.last_predicted_action)

    def action_exploration(
        self, action_function: Callable[[Any], int]
    ) -> Callable[[Any], int]:
        return action_function

    def compute_intrinsic_reward(self, current_state, next_state, action):
        self.last_predicted_action, current_state_embedding = self.inverse_model(
            current_state, next_state
        )
        episodic_reward = self._episodic_reward(current_state_embedding)
        exploration_bonus = self._exploration_bonus(current_state)
        return episodic_reward * th.clip(
            th.Tensor(exploration_bonus), 1, self.exploration_bonus_limit
        )

    def _episodic_reward(
        self, current_embedding: th.Tensor, denominator_constant: float = 1e-5
    ):
        current_embedding_detached = current_embedding.detach().numpy()

        if len(self.memory) <= self.neighbors:
            neighbor_distances = [0]
        else:
            # Compute K nearest neighbors wrt inverse kernel from memory
            neighbors_model = NearestNeighbors(
                n_neighbors=self.neighbors, metric=self._inverse_kernel
            )
            memory_array = np.array(self.memory)
            neighbors_model.fit(memory_array)
            neighbor_distances, _ = neighbors_model.kneighbors(
                np.array(current_embedding_detached).reshape(1, -1),
            )

        # Update Memory
        self.memory.append(current_embedding_detached)

        # Episodic Reward
        return 1 / np.sqrt(np.sum(neighbor_distances) + denominator_constant)

    def _exploration_bonus(self, current_state: dgl.DGLHeteroGraph):
        distiller_error: th.Tensor = self.random_network_distiller(current_state)
        self.distiller_error_running_mean.update(distiller_error.data)
        self.distiller_error_running_standard_deviation.update(distiller_error.data)
        return (
            (
                1
                + (distiller_error - self.distiller_error_running_mean.value)
                / self.distiller_error_running_standard_deviation.value
            )
            if self.distiller_error_running_standard_deviation.value != 0
            else 1
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
            actions: int,
            architecture: InverseModelArchitecture,
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
            # These attributes are used for reflection
            # Do not remove it
            self.embedding_size = self.embedding_network.get_embedding_dimension()
            self.action_space_size = actions

            self.action_prediction_stream: nn.Sequential = get_network(
                self,
                architecture=architecture.action_prediction_stream,
                is_graph_network=False,
            )
            self.loss = nn.CrossEntropyLoss()

            self.optimizer: th.optim.Optimizer = th.optim.Adam(
                self.parameters(), lr=architecture.learning_rate
            )

        def forward(
            self, current_state: dgl.DGLHeteroGraph, next_state: dgl.DGLHeteroGraph
        ) -> Tuple[Tensor, Tensor]:
            # -> (node, embedding_size)
            current_state_node_embedding: th.Tensor = self.embedding_network(
                current_state
            )
            # -> (embedding_size)
            current_state_embedding = th.mean(current_state_node_embedding, dim=0)

            # -> (node, embedding_size)
            next_state_node_embedding: th.Tensor = self.embedding_network(next_state)
            # -> (embedding_size)
            next_state_embedding = th.mean(next_state_node_embedding, dim=0)

            predicted_action = self.action_prediction_stream(
                th.cat((current_state_embedding, next_state_embedding))
            )
            return predicted_action, current_state_embedding

        def learn(self, action: int, predicted_action: th.Tensor):
            self.optimizer.zero_grad()
            loss = self.loss(
                predicted_action.reshape([1, self.action_space_size]),
                th.Tensor([action]).long(),
            )
            loss.backward()
            self.optimizer.step()

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
