import copy
from abc import ABC
from dataclasses import asdict
from random import choice
from typing import Any, Dict, List, Optional, OrderedDict, Tuple

import dgl
import networkx as nx
import psutil
import torch as th
import torch.nn as nn
from pop.agents.exploration.exploration_module import ExplorationModule
from pop.agents.exploration.exploration_module_factory import get_exploration_module
from dgl import DGLHeteroGraph
from grid2op.Observation import BaseObservation
from torch import Tensor

from pop.agents.loggable_module import LoggableModule
from pop.agents.replay_buffer import ReplayMemory, Transition
from pop.configs.agent_architecture import AgentArchitecture
from pop.networks.dueling_net import DuelingNet
from pop.networks.serializable_module import SerializableModule


class BaseGCNAgent(SerializableModule, LoggableModule, ABC):

    # These names are used to find files in the load directory
    # When loading an agent
    target_network_name_suffix: str = "_target_network"
    q_network_name_suffix: str = "_q_network"
    optimizer_class: str = "th.optim.Adam"

    def __init__(
        self,
        agent_actions: Optional[int],
        node_features: Optional[List[str]],
        architecture: Optional[AgentArchitecture],
        training: bool,
        name: str,
        device: str,
        log_dir: Optional[str],
        tensorboard_dir: Optional[str],
        feature_ranges: Dict[str, Tuple[float, float]],
        edge_features: Optional[List[str]] = None,
        single_node_features: Optional[int] = None,
    ):
        SerializableModule.__init__(self, name=name, log_dir=log_dir)
        LoggableModule.__init__(self, tensorboard_dir=tensorboard_dir)

        # Agent Architecture
        self.feature_ranges = feature_ranges
        self.architecture = architecture
        self.actions = agent_actions
        self.node_features_schema: Optional[List[str]] = node_features
        self.edge_features_schema: Optional[List[str]] = edge_features
        self.node_features: int = (
            len(node_features) if single_node_features is None else single_node_features
        )
        self.edge_features: Optional[int] = (
            len(edge_features) if edge_features is not None else None
        )
        self.name = name
        self.single_node_features = single_node_features

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
            node_features=self.node_features,
            edge_features=self.edge_features,
            embedding_architecture=architecture.embedding,
            advantage_stream_architecture=architecture.advantage_stream,
            value_stream_architecture=architecture.value_stream,
            name=name + "_dueling",
            log_dir=None,
            feature_ranges=feature_ranges,
        ).to(self.device)
        self.target_network: DuelingNet = copy.deepcopy(self.q_network).to(self.device)

        # Logging
        self.train_steps: int = 0
        self.episodes: int = 0
        self.alive_steps: int = 0
        self.learning_steps: int = 0

        # Optimizer
        self.optimizer: th.optim.Optimizer = th.optim.Adam(
            self.q_network.parameters(),
            lr=self.architecture.learning_rate,
            eps=self.architecture.adam_epsilon,
        )

        # Huber Loss initialization with delta
        self.loss_func: nn.HuberLoss = nn.HuberLoss(
            delta=self.architecture.huber_loss_delta
        )

        # Replay Buffer
        self.memory: ReplayMemory = ReplayMemory(self.architecture.replay_memory)

        # Training or Evaluation
        self.training: bool = training
        self.last_action: Optional[int] = None

        self.exploration: ExplorationModule = get_exploration_module(self)

    def get_exploration_logs(self) -> Dict[str, Any]:
        return self.exploration.get_state_to_log()

    def get_name(self):
        return self.name

    def set_cpu_affinity(self, cpus: List[int]):
        psutil.Process().cpu_affinity(cpus)

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

    def _take_action(
        self,
        transformed_observation: DGLHeteroGraph,
        mask: Optional[List[int]] = None,
    ) -> int:

        # -> (actions)
        advantages: Tensor = self.q_network.advantage(transformed_observation)
        action = int(th.argmax(advantages).item())
        action = action
        self.last_action = action

        return action

    def q_value(self, transformed_observation: DGLHeteroGraph, action: int):
        return self.q_network(transformed_observation).squeeze()[action].item()

    def take_action(
        self, transformed_observation: DGLHeteroGraph, mask: Optional[List[int]] = None
    ) -> Tuple[int, float]:
        if self.edge_features is not None and transformed_observation.num_edges() == 0:
            transformed_observation.add_edges([0], [0])
            self._add_fake_edge_features(transformed_observation)

        if self.training:
            action = self.exploration.action_exploration(self._take_action)(
                self, transformed_observation, mask=mask
            )
        else:
            action = self._take_action(transformed_observation, mask=mask)

        return action, self.q_value(transformed_observation, action)

    def update_mem(
        self,
        observation: DGLHeteroGraph,
        action: int,
        reward: float,
        next_observation: DGLHeteroGraph,
        done: bool,
    ) -> None:
        self._cast_features_to_float32(observation)
        self._cast_features_to_float32(next_observation)

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
            self.architecture.batch_size
        )
        transitions = Transition(*zip(*transitions))

        loss, td_error = self.compute_loss(
            transitions, th.Tensor(sampling_weights).to(self.device)
        )

        # Backward propagation
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=40)
        self.optimizer.step()

        # Update priorities for sampling
        self.memory.update_priorities(
            memory_indices, td_error.abs().detach().numpy().flatten()
        )
        return loss.item()

    def _add_fake_edge_features(self, graph: dgl.DGLGraph):
        if self.edge_features_schema is not None:
            for edge_feature in self.edge_features_schema:
                graph.edata[edge_feature] = th.zeros((1,)).to(self.device)
        else:
            raise Exception("Called add_fake_edge_features without features")

    def _add_fake_node(self, graph: dgl.DGLGraph):

        graph.add_nodes(1)
        if not self.single_node_features:
            for node_feature in self.node_features_schema:
                graph.ndata[node_feature] = th.zeros((1,)).to(self.device)
        else:
            graph.ndata[self.node_features_schema[0]] = th.zeros(
                1, self.single_node_features
            ).to(self.device)

    def _cast_features_to_float32(self, graph: DGLHeteroGraph):
        for feature in graph.node_attr_schemes().keys():
            graph.ndata[feature] = graph.ndata[feature].type(th.float32)
        for feature in graph.edge_attr_schemes().keys():
            graph.edata[feature] = graph.edata[feature].type(th.float32)

    def _step(
        self,
        observation: DGLHeteroGraph,
        action: int,
        reward: float,
        next_observation: DGLHeteroGraph,
        done: bool,
        stop_decay: bool = False,
    ) -> Tuple[Optional[float], float]:
        # This method is redefined by ExplorationModule.apply_intrinsic_reward() at runtime if training=True

        if done:
            self.episodes += 1
            self.alive_steps = 0

        else:
            self.alive_steps += 1

        self.memory.push(observation, action, next_observation, reward, done)

        if not stop_decay and self.training:
            self.exploration.update(action)
            self.memory.update()

        # every so often the agents should learn from experiences
        if self.train_steps % self.architecture.learning_frequency == 0:
            loss = self.learn()
            if loss is not None:
                self.learning_steps += 1
                return loss, reward
        return None, reward

    def step(
        self,
        observation: DGLHeteroGraph,
        action: int,
        reward: float,
        next_observation: DGLHeteroGraph,
        done: bool,
        stop_decay: bool = False,
    ) -> Tuple[Optional[float], float]:

        self._cast_features_to_float32(observation)
        self._cast_features_to_float32(next_observation)

        self.train_steps += 1

        if observation.num_nodes() == 0:
            self._add_fake_node(observation)

        if next_observation.num_nodes() == 0:
            self._add_fake_node(next_observation)

        if self.edge_features is not None:
            if observation.num_edges() == 0:
                observation.add_edges([0], [0])
                self._add_fake_edge_features(observation)
            if next_observation.num_edges() == 0:
                next_observation.add_edges([0], [0])
                self._add_fake_edge_features(next_observation)

        if self.training:
            return self.exploration.apply_intrinsic_reward(
                self._step, self.architecture.intrinsic_reward_relevance
            )(
                self,
                observation,
                action,
                reward,
                next_observation,
                done,
                stop_decay=stop_decay,
            )

        else:
            return self._step(
                observation,
                action,
                reward,
                next_observation,
                done,
                stop_decay=stop_decay,
            )

    def get_state(
        self,
    ) -> Dict[str, Any]:
        return {
            "optimizer_state": self.optimizer.state_dict(),
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "memory": self.memory.get_state(),
            "exploration": self.exploration.get_state(),
            "alive_steps": self.alive_steps,
            "train_steps": self.train_steps,
            "learning_steps": self.learning_steps,
            "agent_actions": self.actions,
            "node_features": self.node_features_schema,
            "edge_features": self.edge_features_schema,
            "single_node_features": self.single_node_features,
            "architecture": asdict(self.architecture),
            "name": self.name,
            "training": self.training,
            "device": self.device,
            "feature_ranges": self.feature_ranges,
        }

    def load_state(
        self,
        optimizer_state: dict,
        q_network_state: OrderedDict[str, Tensor],
        target_network_state: OrderedDict[str, Tensor],
        memory: dict,
        exploration: dict,
        alive_steps: int,
        train_steps: int,
        learning_steps: int,
        reset_exploration: bool = False,
    ) -> None:
        self.alive_steps = alive_steps
        self.train_steps = train_steps
        self.learning_steps = learning_steps
        self.optimizer.load_state_dict(optimizer_state)
        self.q_network.load_state_dict(q_network_state)
        self.target_network.load_state_dict(target_network_state)
        self.memory.load_state(memory)
        if not reset_exploration:
            self.exploration.load_state(exploration)

    @staticmethod
    def batch_observations(graphs: List[DGLHeteroGraph]) -> DGLHeteroGraph:
        return dgl.batch(graphs)

    @staticmethod
    def from_networkx_to_dgl(
        graph: nx.Graph, node_features: List[str], edge_features: List[str], device: str
    ) -> DGLHeteroGraph:
        return dgl.from_networkx(
            graph.to_directed(),
            node_attrs=node_features if len(graph.nodes) > 0 else [],
            edge_attrs=edge_features if len(graph.edges) > 0 else [],
            device=device,
        )
