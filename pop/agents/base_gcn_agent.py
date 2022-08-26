from abc import ABC
from dataclasses import asdict
from typing import Optional, Tuple, List, Dict, Any, OrderedDict

import networkx as nx
from dgl import DGLHeteroGraph
from grid2op.Observation import BaseObservation
from torch import Tensor

from pop.agents.loggable_module import LoggableModule
from pop.agents.replay_buffer import ReplayMemory, Transition
from pop.configs.agent_architecture import AgentArchitecture
from pop.networks.serializable_module import SerializableModule
from pop.networks.dueling_net import DuelingNet
import copy
import numpy as np
import torch as th
import torch.nn as nn
import dgl
from random import choice
import psutil


class BaseGCNAgent(SerializableModule, LoggableModule, ABC):

    # These names are used to find files in the load directory
    # When loading an agent
    target_network_name_suffix: str = "_target_network"
    q_network_name_suffix: str = "_q_network"
    optimizer_class: str = "th.optim.Adam"

    def __init__(
        self,
        agent_actions: Optional[int],
        node_features: Optional[int],
        architecture: Optional[AgentArchitecture],
        training: bool,
        name: str,
        device: str,
        log_dir: Optional[str],
        tensorboard_dir: Optional[str],
        edge_features: Optional[int] = None,
    ):
        SerializableModule.__init__(self, name=name, log_dir=log_dir)
        LoggableModule.__init__(self, tensorboard_dir=tensorboard_dir)

        # Agent Architecture
        self.architecture = architecture
        self.actions = agent_actions
        self.node_features = node_features
        self.edge_features = edge_features
        self.name = name

        # Initialize Torch device
        if device is None:
            self.device: th.device = th.device(
                "cuda:0" if th.cuda.is_available() else "cpu"
            )
        else:
            self.device: th.device = th.device(device)

        if architecture is None:
            return

        # Initialize deep networks
        self.q_network: DuelingNet = DuelingNet(
            action_space_size=agent_actions,
            node_features=node_features,
            edge_features=edge_features,
            embedding_architecture=architecture.embedding,
            advantage_stream_architecture=architecture.advantage_stream,
            value_stream_architecture=architecture.value_stream,
            name=name + "_dueling",
            log_dir=None,
        )
        self.target_network: DuelingNet = copy.deepcopy(self.q_network)

        # Logging
        self.decay_steps: int = 0
        self.train_steps: int = 0
        self.episodes: int = 0
        self.alive_steps: int = 0
        self.learning_steps: int = 0
        self.epsilon: float = self.architecture.exploration.max_epsilon

        # Optimizer
        self.optimizer: th.optim.Optimizer = th.optim.Adam(
            self.q_network.parameters(), lr=self.architecture.learning_rate
        )

        # Huber Loss initialization with delta
        self.loss_func: nn.HuberLoss = nn.HuberLoss(
            delta=self.architecture.huber_loss_delta
        )

        # Replay Buffer
        self.memory: ReplayMemory = ReplayMemory(
            int(self.architecture.replay_memory.capacity),
            self.architecture.replay_memory.alpha,
        )

        # Training or Evaluation
        self.training: bool = training
        self.last_action: Optional[int] = None

    def get_name(self):
        return self.name

    def compute_loss(
        self, transitions_batch: Transition, sampling_weights: Tensor
    ) -> Tuple[Tensor, Tensor]:

        observation_batch = self.batch_observations(transitions_batch.observation)
        next_observation_batch = self.batch_observations(
            transitions_batch.next_observation
        )

        # Get 1 action per batch and restructure as an index for gather()
        # -> (batch_size)
        actions = (
            th.Tensor(transitions_batch.action)
            .unsqueeze(1)
            .type(th.int64)
            .to(self.device)
        )

        # Get rewards and unsqueeze to get 1 reward per batch
        # -> (batch_size)
        rewards = th.Tensor(transitions_batch.reward).unsqueeze(1).to(self.device)

        # Compute Q value for the current observation
        # -> (batch_size)
        q_values: Tensor = (
            self.q_network(observation_batch).gather(1, actions).to(self.device)
        )

        # Compute TD error
        # -> (batch_size, action_space_size)
        target_q_values: Tensor = self.target_network(next_observation_batch).to(
            self.device
        )

        # -> (batch_size)
        best_actions: Tensor = (
            th.argmax(self.q_network(next_observation_batch), dim=1)
            .unsqueeze(1)
            .type(th.int64)
        ).to(self.device)

        # -> (batch_size)
        td_errors: Tensor = rewards + self.architecture.gamma * target_q_values.gather(
            1, best_actions
        ).to(self.device)

        # deltas = weights (q_values - td_errors)
        # to keep interfaces general we distribute weights
        # -> (1)
        loss: Tensor = self.loss_func(
            q_values * sampling_weights, td_errors * sampling_weights
        ).to(self.device)

        return loss, td_errors

    def get_memory(self) -> ReplayMemory:
        return self.memory

    def _exponential_decay(self, max_val: float, min_val: float, decay: int) -> float:
        return min_val + (max_val - min_val) * np.exp(-1.0 * self.decay_steps / decay)

    def take_action(self, transformed_observation: DGLHeteroGraph, *args) -> int:
        if len(args) > 0:
            psutil.Process().cpu_affinity(args[0])

        self.epsilon = self._exponential_decay(
            self.architecture.exploration.max_epsilon,
            self.architecture.exploration.min_epsilon,
            self.architecture.exploration.epsilon_decay,
        )
        action_list = list(range(self.actions))
        if self.training:
            # epsilon-greedy Exploration
            if np.random.rand() <= self.epsilon:
                self.last_action = None
                return np.random.choice(action_list)

        # -> (actions)
        advantages: Tensor = self.q_network.advantage(transformed_observation)
        action = int(th.argmax(advantages).item())
        action = action if action != self.last_action else 0
        self.last_action = action

        return action

    def update_mem(
        self,
        observation: DGLHeteroGraph,
        action: int,
        reward: float,
        next_observation: DGLHeteroGraph,
        done: bool,
    ) -> None:

        self.memory.push(observation, action, next_observation, reward, done)

    def learn(self) -> Optional[float]:
        if len(self.memory) < self.architecture.batch_size:
            return None
        if (
            self.train_steps % self.architecture.target_network_weight_replace_steps
            == 0
        ):
            self.target_network.parameters = self.q_network.parameters

        # Sample from Replay Memory and unpack
        memory_indices, transitions, sampling_weights = self.memory.sample(
            self.architecture.batch_size,
            self._exponential_decay(
                self.architecture.replay_memory.max_beta,
                self.architecture.replay_memory.min_beta,
                self.architecture.replay_memory.beta_decay,
            ),
        )
        transitions = Transition(*zip(*transitions))

        loss, td_error = self.compute_loss(transitions, th.Tensor(sampling_weights))

        # Backward propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities for sampling
        self.memory.update_priorities(
            memory_indices, td_error.abs().detach().numpy().flatten()
        )
        return loss.item()

    def step(
        self,
        observation: DGLHeteroGraph,
        action: int,
        reward: float,
        next_observation: DGLHeteroGraph,
        done: bool,
        stop_decay: bool = False,
    ) -> Optional[float]:

        self.train_steps += 1
        if done:
            self.episodes += 1
            self.alive_steps = 0

        else:
            self.memory.push(observation, action, next_observation, reward, done)
            self.alive_steps += 1

            if not stop_decay and self.training:
                self.decay_steps += 1

            # every so often the agents should learn from experiences
            if self.train_steps % self.architecture.learning_frequency == 0:
                loss: Optional[float] = self.learn()
                self.learning_steps += 1
                return loss
            return None

    def get_state(
        self,
    ) -> Dict[str, Any]:
        return {
            "optimizer_state": self.optimizer.state_dict(),
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "memory": self.memory.get_state(),
            "decay_steps": self.decay_steps,
            "alive_steps": self.alive_steps,
            "train_steps": self.train_steps,
            "learning_steps": self.learning_steps,
            "agent_actions": self.actions,
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "architecture": asdict(self.architecture),
            "name": self.name,
            "training": self.training,
            "device": self.device,
        }

    def load_state(
        self,
        optimizer_state: dict,
        q_network_state: OrderedDict[str, Tensor],
        target_network_state: OrderedDict[str, Tensor],
        memory: dict,
        decay_steps: int,
        alive_steps: int,
        train_steps: int,
        learning_steps: int,
    ) -> None:
        self.decay_steps = decay_steps
        self.alive_steps = alive_steps
        self.train_steps = train_steps
        self.learning_steps = learning_steps
        self.optimizer.load_state_dict(optimizer_state)
        self.q_network.load_state_dict(q_network_state)
        self.target_network.load_state_dict(target_network_state)
        self.memory.load(memory)

    @staticmethod
    def batch_observations(graphs: List[DGLHeteroGraph]) -> DGLHeteroGraph:
        return dgl.batch(graphs)

    @staticmethod
    def to_dgl(obs: BaseObservation, device: str) -> DGLHeteroGraph:

        # Convert Grid2Op graph to a directed (for compatibility reasons) networkx graph
        net = obs.as_networkx()
        net = net.to_directed()  # Typing error from networkx, ignore it

        # Convert from networkx to dgl graph
        return BaseGCNAgent.from_networkx_to_dgl(net, device)

    @staticmethod
    def from_networkx_to_dgl(graph: nx.Graph, device: str) -> DGLHeteroGraph:
        return dgl.from_networkx(
            graph.to_directed(),
            node_attrs=list(graph.nodes[choice(list(graph.nodes))].keys())
            if len(graph.nodes) > 0
            else [],
            edge_attrs=list(graph.edges[choice(list(graph.edges))].keys())
            if len(graph.edges) > 0
            else [],
            device=device,
        )
