import json
from abc import ABC
from typing import Union, Optional, Tuple

from dgl import DGLHeteroGraph
from torch import Tensor

from node_agents.utilities import batch_observations
from networks.dueling_net import DuelingNet
from pop.dueling_networks.dueling_net_factory import get_dueling_net
import networkx as nx
import numpy as np
import torch as th
import torch.nn as nn
from pop.node_agents.replay_buffer import ReplayMemory, Transition


class BaseGCNAgent(ABC):

    # This names are used to find files in the load directory
    # When loading an agent
    target_network_name_suffix: str = "_target_network"
    q_network_name_suffix: str = "_q_network"
    optimizer_class: str = "th.optim.Adam"

    def __init__(
        self,
        agent_actions: int,
        node_features: int,
        edge_features: int,
        architecture: Union[str, dict],
        name: str,
        training: bool,
        device: str,
        **kwargs
    ):

        # Agent Architecture
        self.architecture = (
            json.load(open(architecture)) if type(architecture) is str else architecture
        )
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

        # Initialize deep networks
        self.q_network: DuelingNet = get_dueling_net(
            node_features=node_features,
            edge_features=edge_features,
            action_space_size=agent_actions,
            architecture=self.architecture["network"],
            name=name + self.q_network_name_suffix,
        ).to(self.device)
        self.target_network: DuelingNet = get_dueling_net(
            node_features=node_features,
            edge_features=edge_features,
            action_space_size=agent_actions,
            architecture=self.architecture["network"],
            name=name + self.target_network_name_suffix,
        ).to(self.device)

        # Optimizer
        self.optimizer: th.optim.Optimizer = th.optim.Adam(
            self.q_network.parameters(), lr=self.architecture["learning_rate"]
        )

        # Huber Loss initialization with delta
        self.loss_func: nn.HuberLoss = nn.HuberLoss(delta=self.architecture["delta"])

        # Replay Buffer
        self.memory: ReplayMemory = ReplayMemory(int(1e5), self.architecture["alpha"])

        # Reporting
        self.trainsteps: int = 0
        self.decay_steps: int = 0
        self.episodes: int = 0
        self.alive_steps: int = 0
        self.learning_steps: int = 0
        self.cumulative_reward: float = 0

        # Training or Evaluation
        self.training: bool = training

    def compute_loss(
        self, transitions_batch: Transition, sampling_weights: Tensor
    ) -> Tuple[Tensor, Tensor, DGLHeteroGraph, DGLHeteroGraph]:

        # Unwrap batch
        # Get observation start and end

        observation_batch = batch_observations(
            transitions_batch.observation, self.device
        )
        next_observation_batch = batch_observations(
            transitions_batch.next_observation, self.device
        )
        # Get 1 action per batch and restructure as an index for gather()
        actions = (
            th.Tensor(transitions_batch.action)
            .unsqueeze(1)
            .type(th.int64)
            .to(self.device)
        )

        # Get rewards and unsqueeze to get 1 reward per batch
        rewards = th.Tensor(transitions_batch.reward).unsqueeze(1).to(self.device)

        # Compute Q value for the current observation
        q_values: Tensor = (
            self.q_network(observation_batch).gather(1, actions).to(self.device)
        )

        # Compute TD error
        target_q_values: Tensor = self.target_network(next_observation_batch).to(
            self.device
        )
        best_actions: Tensor = (
            th.argmax(self.q_network(next_observation_batch), dim=1)
            .unsqueeze(1)
            .type(th.int64)
        ).to(self.device)
        td_errors: Tensor = rewards + self.architecture[
            "gamma"
        ] * target_q_values.gather(1, best_actions).to(self.device)

        # deltas = weights (q_values - td_errors)
        # to keep interfaces general we distribute weights
        loss: Tensor = self.loss_func(
            q_values * sampling_weights, td_errors * sampling_weights
        ).to(self.device)

        return loss, td_errors, observation_batch, next_observation_batch

    def exponential_decay(self, max_val: float, min_val: float, decay: int) -> float:
        return min_val + (max_val - min_val) * np.exp(-1.0 * self.decay_steps / decay)

    def take_action(
        self,
        transformed_observation: DGLHeteroGraph,
    ) -> Tuple[int, float]:

        epsilon = self.exponential_decay(
            self.architecture["max_epsilon"],
            self.architecture["min_epsilon"],
            self.architecture["epsilon_decay"],
        )
        if self.training and not self.architecture["network"].get("noisy_layers"):
            # epsilon-greedy Exploration
            if np.random.rand() <= epsilon:
                return np.random.choice(list(range(self.actions))), epsilon

        # Exploitation or Noisy Layers Exploration
        graph = transformed_observation  # extract output of converted obs

        advantages: Tensor = self.q_network.advantage(graph.to(self.device))

        return (
            int(th.argmax(advantages).item()),
            epsilon,
        )

    def update_mem(
        self,
        observation: DGLHeteroGraph,
        action: int,
        reward: float,
        next_observation: DGLHeteroGraph,
        done: bool,
    ) -> None:

        self.memory.push(observation, action, next_observation, reward, done)

    def learn(self) -> Optional[Tensor]:
        if len(self.memory) < self.architecture["batch_size"]:
            return None
        if self.trainsteps % self.architecture["replace"] == 0:
            self.target_network.parameters = self.q_network.parameters

        # Sample from Replay Memory and unpack
        idxs, transitions, sampling_weights = self.memory.sample(
            self.architecture["batch_size"],
            self.exponential_decay(
                self.architecture["max_beta"],
                self.architecture["min_beta"],
                self.architecture["beta_decay"],
            ),
        )
        transitions = Transition(*zip(*transitions))

        (
            loss,
            td_error,
            observation_batch,
            next_observation_batch,
        ) = self.compute_loss(transitions, th.Tensor(sampling_weights))

        # Backward propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities for sampling
        self.memory.update_priorities(idxs, td_error.abs().detach().numpy().flatten())

        return loss

    def step(
        self,
        observation,
        action: int,
        reward: float,
        next_observation: nx.Graph,
        done: bool,
        stop_decay: bool = False,
    ) -> Tuple[Optional[Tensor], Optional[dict], Optional[dict]]:

        if done:
            self.episodes += 1
            self.trainsteps += 1

        else:
            self.memory.push(observation, action, next_observation, reward, done)
            self.trainsteps += 1
            self.alive_steps += 1

            if not stop_decay and self.training:
                self.decay_steps += 1

            # every so often the node_agents should learn from experiences
            if self.trainsteps % self.architecture["learning_frequency"] == 0:
                loss = self.learn()
                self.learning_steps += 1
                return (
                    loss,
                    self.q_network.state_dict(),
                    self.target_network.state_dict(),
                )
        return None, None, None
