from abc import ABC
from typing import Optional, Tuple, List

from dgl import DGLHeteroGraph
from torch import Tensor

from configs.agent_architecture import AgentArchitecture
from networks.serializable_module import SerializableModule
from agents.loggable_module import LoggableModule
from agents.utilities import batch_observations
from networks.dueling_net import DuelingNet
import networkx as nx
import copy
import numpy as np
import numpy.ma as ma
import torch as th
import torch.nn as nn
from pop.agents.replay_buffer import ReplayMemory, Transition


class BaseGCNAgent(ABC, SerializableModule, LoggableModule):

    # This names are used to find files in the load directory
    # When loading an agent
    target_network_name_suffix: str = "_target_network"
    q_network_name_suffix: str = "_q_network"
    optimizer_class: str = "th.optim.Adam"

    def __init__(
        self,
        agent_actions: int,
        node_features: int,
        architecture: AgentArchitecture,
        name: str,
        training: bool,
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

        # Reporting
        self.decay_steps: int = 0
        self.train_steps: int = 0
        self.episodes: int = 0
        self.alive_steps: int = 0
        self.learning_steps: int = 0

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
            int(1e5), self.architecture.replay_memory.alpha
        )

        # Training or Evaluation
        self.training: bool = training

    def compute_loss(
        self, transitions_batch: Transition, sampling_weights: Tensor
    ) -> Tuple[Tensor, Tensor, DGLHeteroGraph, DGLHeteroGraph]:

        observation_batch = batch_observations(
            transitions_batch.observation, self.device
        )
        next_observation_batch = batch_observations(
            transitions_batch.next_observation, self.device
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

        return loss, td_errors, observation_batch, next_observation_batch

    def exponential_decay(self, max_val: float, min_val: float, decay: int) -> float:
        return min_val + (max_val - min_val) * np.exp(-1.0 * self.decay_steps / decay)

    def take_action(
        self,
        transformed_observation: DGLHeteroGraph,
        mask: Optional[List[int]] = None,
    ) -> Tuple[int, float]:

        epsilon = self.exponential_decay(
            self.architecture.exploration.max_epsilon,
            self.architecture.exploration.min_epsilon,
            self.architecture.exploration.epsilon_decay,
        )
        action_list = list(range(self.actions))
        if self.training:
            # epsilon-greedy Exploration
            if np.random.rand() <= epsilon:
                if mask is None:
                    return (
                        np.random.choice(action_list),
                        epsilon,
                    )
                else:
                    return (
                        np.random.choice(
                            action_list,
                            p=[
                                1 / len(mask) if action in mask else 0
                                for action in action_list
                            ],
                        ),
                        epsilon,
                    )

        # -> (actions)
        advantages: Tensor = self.q_network.advantage(transformed_observation)
        if mask is not None:
            advantages[
                [False if action in mask else True for action in action_list]
            ] = float("-inf")

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
        if len(self.memory) < self.architecture.batch_size:
            return None
        if (
            self.train_steps % self.architecture.target_network_weight_replace_steps
            == 0
        ):
            self.target_network.parameters = self.q_network.parameters

        # Sample from Replay Memory and unpack
        idxs, transitions, sampling_weights = self.memory.sample(
            self.architecture.batch_size,
            self.exponential_decay(
                self.architecture.replay_memory.max_beta,
                self.architecture.replay_memory.min_beta,
                self.architecture.replay_memory.beta_decay,
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
    ) -> Optional[Tuple[Tensor, dict, dict]]:

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
                loss = self.learn()
                self.learning_steps += 1
                return (
                    loss,
                    self.q_network.state_dict(),
                    self.target_network.state_dict(),
                )
