from pathlib import Path
import json
import networkx as nx

from dgl import DGLHeteroGraph
from tqdm import tqdm
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
from grid2op.Action import BaseAction, ActionSpace
from grid2op.Observation import BaseObservation
from grid2op.Environment import BaseEnv
import torch as th
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
import dgl
import numpy as np

from pop.dueling_networks.dueling_net import DuelingNet
from pop.dueling_networks.dueling_net_factory import get_dueling_net
from pop.node_agents.replay_buffer import ReplayMemory, Transition

from typing import List, Optional, Union, Tuple

from pop.node_agents.utilities import to_dgl, batch_observations


class DoubleDuelingGCNAgent(AgentWithConverter):
    """
    Double Dueling Graph Convolutional Neural (GCN) Network Agent.
    GCN is used to embed the graph.
    In the Dueling framework the Neural Network predicts:
    - the Value function for the current observation;
    - the Advantage values for each action over the current observation.

    The Q values are then computed by aggregating Value function and Advantage values.
    See :class:`DuelingGCN` for more details.

    In a double GCN two equal models are used:
    - the Q Network is updated at each timestep;
    - the Target Network is updated every :attribute:`replace` steps to avoid bias issues.

    """

    # This names are used to find files in the load directory
    # When loading an agent
    target_network_name_suffix: str = "_target_network"
    q_network_name_suffix: str = "_q_network"
    optimizer_class: str = "th.optim.Adam"

    def __init__(
        self,
        agent_actions: List[BaseAction],
        full_action_space: ActionSpace,
        node_features: int,
        edge_features: int,
        architecture: Union[str, dict],
        name: str,
        seed: int,
        training: bool,
        tensorboard_log_dir: str,
        log_dir,
        device: Optional[str] = None,
    ):
        self.architecture: dict = (
            json.load(open(architecture)) if type(architecture) is str else architecture
        )
        super().__init__(
            full_action_space,
            IdToAct,
            all_actions=agent_actions,
            **self.architecture["converter_kwargs"]
        )

        self.name = name
        self.action_space_converter = IdToAct(full_action_space)
        self.action_space_converter.init_converter(all_actions=agent_actions)
        self.action_space_converter.seed(seed)
        self.node_features = node_features
        self.edge_features = edge_features
        self.seed = seed
        self.tensorboard_log_dir = tensorboard_log_dir
        self.log_dir = log_dir
        self.loss: Tensor = th.Tensor()

        # Initialize Torch device
        if device is None:
            self.device: th.device = th.device(
                "cuda:0" if th.cuda.is_available() else "cpu"
            )
        else:
            self.device: th.device = th.device(device)

        self.name = name

        # Initialize deep networks
        self.q_network: DuelingNet = get_dueling_net(
            node_features=node_features,
            edge_features=edge_features,
            action_space_size=len(agent_actions),
            architecture=self.architecture["network"],
            name=name + self.q_network_name_suffix,
            log_dir=log_dir,
        ).to(self.device)
        self.target_network: DuelingNet = get_dueling_net(
            node_features=node_features,
            edge_features=edge_features,
            action_space_size=len(agent_actions),
            architecture=self.architecture["network"],
            name=name + self.target_network_name_suffix,
            log_dir=log_dir,
        ).to(self.device)

        # Optimizer
        self.optimizer: th.optim.Optimizer = th.optim.Adam(
            self.q_network.parameters(), lr=self.architecture["learning_rate"]
        )

        # Replay Buffer
        self.memory: ReplayMemory = ReplayMemory(int(1e5), self.architecture["alpha"])

        # Reporting
        self.trainsteps: int = 0
        self.episodes: int = 0
        self.alive_steps: int = 0
        self.alive_incremental_mean: float = 0
        self.learning_steps: int = 0
        self.reward_incremental_mean: float = 0
        self.cumulative_reward: float = 0

        # Logging
        self.log_dir: str = log_dir
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.log_file: str = str(Path(self.log_dir, name + ".pt"))

        # Huber Loss initialization with delta
        self.loss_func: nn.HuberLoss = nn.HuberLoss(delta=self.architecture["delta"])

        # Training or Evaluation
        self.training: bool = training
        if training:

            # Tensorboard
            Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
            self.writer: Optional[SummaryWriter]
            if tensorboard_log_dir is not None:
                self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
            else:
                self.writer = None

    def compute_loss(
        self, transitions_batch: Transition, sampling_weights: th.Tensor
    ) -> Tuple[Tensor, Tensor, DGLHeteroGraph, DGLHeteroGraph]:
        """
        Computes the current loss given a batch of transitions and the sampling weights from
        the prioritized experience replay buffer.
        Parameters
        ----------
        transitions_batch: :class:`Transition`
            A transition is a tuple (observation, next_observation, action, reward, done).
            A batch of transitions is only a transition which has multiple values for each tuple entry.

        sampling_weights: :class:`th.Tensor`
            Sampling weights are associated to each transition sampled from the prioritized experience replay buffer
        """

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
        """
        Exponential decay schedule, value evolves with trainsteps stored in the node_agents's state

        Parameters
        ----------
        max_val: ``float``
            upperbound of the returned value, starting point of the decay
        min_val: ``float``
            lowerbound of the returned value, ending point of the decay
        decay: ``int``
            decay parameter, the higher the faster the decay
        Return
        ------
        current_decay_value: ``float``
            current decay value computed over max, min, decay speed and number of elapsed trainsteps
        """
        return min_val + (max_val - min_val) * np.exp(-1.0 * self.trainsteps / decay)

    def my_act(
        self,
        transformed_observation: dgl.DGLHeteroGraph,
        reward=None,
        done=False,
    ) -> int:
        """
        Action function, maps an observation to an action.
        Overrides "my_act" from :class:`AgentWithConverter`

        Parameters
        ----------
        transformed_observation: :class:`dgl.DGLHeteroGraph`
            observation of the environment already transformed through
            the :class:`AgentWithConverter` into a Deep Graph Library Graph Representation with node and edge features
        reward: ``float``
            Unused for :class:`AgentWithConverter` compatibility
        done: ``done``
            Unused for :class:`AgentWithConverter` compatibility

        Return
        ------
        action: ``int``
            Action encoded as an integer, to be used with :class:`IdtoAct` converter from grid2op to go back to
            a verbose textual representation of the action on the grid
        """

        if self.training:
            # epsilon-greedy Exploration
            if np.random.rand() <= self.exponential_decay(
                self.architecture["max_epsilon"],
                self.architecture["min_epsilon"],
                self.architecture["epsilon_decay"],
            ):
                return self.action_space.sample()

        # Exploitation
        graph = transformed_observation  # extract output of converted obs

        advantages: Tensor = self.q_network.advantage(graph.to(self.device))

        return int(th.argmax(advantages).item())

    def update_mem(
        self,
        observation: BaseObservation,
        action: int,
        reward: float,
        next_observation: BaseObservation,
        done: bool,
    ) -> None:
        """
        Add transition to the experience replay buffer

        Parameters
        ----------
        observation: :class:`BaseObservation`
            current observation
        action: int
            action taken from the current observation that produces next_observation
        reward: float
            reward obtained by taking action in the state observed in observation
        next_observation: :class:`BaseObservation`
            observation obtained by taking action in observation
        done: ``bool``
            true if the episode is over
        """

        self.memory.push(observation, action, next_observation, reward, done)

    def convert_obs(self, observation: BaseObservation) -> dgl.DGLHeteroGraph:
        """
        Convert observation to a Deep Graph Library graph.
        Overrides 'convert_obs' from :class:`AgentWithConverter`

        Parameters
        ----------
        observation: :class:``BaseObservation``
            observation to convert

        Return
        ------
        converted_observation: :class:``dgl.DGLHeteroGraph``
            observation converted to a Deep Graph Library graph
        """

        return to_dgl(observation, self.device)

    def save_to_tensorboard(
        self,
        loss: float,
        reward: float,
    ) -> None:
        if self.writer is None:
            print("Warning: trying to save to tensorboard but its deactivated")
            return
        self.writer.add_scalar("Loss/train", loss, self.trainsteps)

        self.writer.add_scalar("Mean_Reward_Over_Batch/train", reward, self.trainsteps)

        self.cumulative_reward += reward

        self.writer.add_scalar(
            "Cumulative_Reward/train", self.cumulative_reward, self.trainsteps
        )

        self.reward_incremental_mean = (
            self.reward_incremental_mean * self.learning_steps + reward
        ) / (self.learning_steps + 1)

        self.writer.add_scalar(
            "Mean_Reward_Over_Learning_Steps/train",
            self.reward_incremental_mean,
            self.learning_steps,
        )
        self.writer.add_scalar(
            "Epsilon/train",
            self.exponential_decay(
                self.architecture["max_epsilon"],
                self.architecture["min_epsilon"],
                self.architecture["epsilon_decay"],
            ),
            self.trainsteps,
        )
        self.writer.add_scalar(
            "Beta/train",
            self.exponential_decay(
                self.architecture["max_beta"],
                self.architecture["min_beta"],
                self.architecture["beta_decay"],
            ),
            self.trainsteps,
        )

    def learn(self) -> None:
        """
        Learning step for the node_agents.
        Learing starts after we have at least 'batch' transitions in memory.
        Every 'replace' steps update the target network.
        Experiences are sampled through prioritized experience replay dependent on exponentially decaying beta.
        Loss is computed through Double Dueling Q learning.

        """
        if len(self.memory) < self.architecture["batch_size"]:
            return
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
            self.loss,
            td_error,
            observation_batch,
            next_observation_batch,
        ) = self.compute_loss(transitions, th.Tensor(sampling_weights))

        # Backward propagation
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

        # Update priorities for sampling
        self.memory.update_priorities(idxs, td_error.abs().detach().numpy().flatten())

        # Save current information to Tensorboard and Checkpoint optimizer and models
        self.save(
            float(self.loss.mean().item()),
            transitions.reward,
        )

    def save(
        self,
        loss: float,
        rewards: List[float],
    ) -> None:
        checkpoint = {
            "optimizer_state": self.optimizer.state_dict(),
            "trainsteps": self.trainsteps,
            "episodes": self.episodes,
            "name": self.name,
            "architecture": self.architecture,
            "tensorboard_log_dir": self.tensorboard_log_dir,
            "log_dir": self.log_dir,
            "seed": self.seed,
            "node_features": self.node_features,
            "edge_features": self.edge_features,
        }
        self.q_network.save()
        self.target_network.save()
        th.save(checkpoint, self.log_file)
        self.save_to_tensorboard(loss, np.mean(rewards))

    @classmethod
    def load(
        cls,
        load_file: str,
        agent_actions: List[BaseAction],
        full_action_space: ActionSpace,
        device: Union[str, th.device],
        training: bool,
    ):
        """
        Load model from checkpoint

        """

        checkpoint = th.load(load_file)

        target_checkpoint = th.load(Path(load_file, cls.target_network_name_suffix))
        q_checkpoint = th.load(Path(load_file, cls.q_network_name_suffix))

        q_net = get_dueling_net(
            name=q_checkpoint["name"],
            architecture=q_checkpoint["architecture"],
            node_features=q_checkpoint["node_features"],
            edge_features=q_checkpoint["edge_features"],
            action_space_size=q_checkpoint["action_space_size"],
            log_dir=Path(q_checkpoint).parent[0],
        ).to(device)

        target_net = get_dueling_net(
            name=target_checkpoint["name"],
            architecture=target_checkpoint["architecture"],
            node_features=target_checkpoint["node_features"],
            edge_features=target_checkpoint["edge_features"],
            action_space_size=target_checkpoint["action_space_size"],
            log_dir=Path(target_checkpoint).parent[0],
        ).to(device)

        agent = DoubleDuelingGCNAgent(
            agent_actions=agent_actions,
            full_action_space=full_action_space,
            node_features=checkpoint["node_features"],
            edge_features=checkpoint["edge_features"],
            architecture=checkpoint["architecture"],
            name=checkpoint["name"],
            seed=checkpoint["seed"],
            training=training,
            tensorboard_log_dir=checkpoint["tensorboard_log_dir"],
            log_dir=checkpoint["log_dir"],
        )

        agent.q_network = q_net
        agent.target_network = target_net
        return agent

    def step(
        self,
        observation: Union[BaseObservation, nx.Graph],
        action: int,
        reward: float,
        next_observation: Union[BaseObservation, nx.Graph],
        done: bool,
    ) -> None:
        """
        Updates the node_agents's state based on feedback received from the environment.

        Parameters:
        -----------
        observation: :class:`BaseObservation`
            previous observation from the environment
        action: ``int``
            the action taken by the node_agents in the previous state.
        reward: ``float``
            the reward received from the environment.
        next_observation: :class:`BaseObservation`
            the resulting state of the environment following the action.
        done: ``bool``
            True if the training episode is over, false otherwise.

        """

        if done:
            if self.writer is not None:
                self.writer.add_scalar(
                    "Steps_Alive_Per_Episode/train",
                    self.alive_steps,
                    self.episodes,
                )

                self.alive_incremental_mean = (
                    self.alive_incremental_mean * self.episodes + self.alive_steps
                ) / (self.episodes + 1)
                self.writer.add_scalar(
                    "Mean_Steps_Alive_Per_Episode/train",
                    self.alive_incremental_mean,
                    self.episodes,
                )

            self.episodes += 1
            self.alive_steps = 0
            self.trainsteps += 1

        else:
            self.memory.push(observation, action, next_observation, reward, done)
            self.trainsteps += 1
            self.alive_steps += 1

            # every so often the node_agents should learn from experiences
            if self.trainsteps % self.architecture["learning_frequency"] == 0:
                self.learn()
                self.learning_steps += 1


def train(
    env: BaseEnv,
    iterations: int,
    agent: DoubleDuelingGCNAgent,
):
    training_step: int = 0
    obs: BaseObservation = (
        env.reset()
    )  # Typing issue for env.reset(), returns BaseObservation
    done = False
    reward = env.reward_range[0]
    total_episodes = len(env.chronics_handler.subpaths)
    with tqdm(total=iterations - training_step) as pbar:
        while training_step < iterations:
            if agent.episodes % total_episodes == 0:
                env.chronics_handler.shuffle()
            if done:
                obs = env.reset()
            encoded_action = agent.my_act(agent.convert_obs(obs), reward, done)
            action = agent.convert_act(encoded_action)
            next_obs, reward, done, _ = env.step(action)
            agent.step(
                observation=obs,
                action=encoded_action,
                reward=reward,
                next_observation=next_obs,
                done=done,
            )
            obs = next_obs
            training_step += 1
            pbar.update(1)
