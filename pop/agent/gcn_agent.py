from pathlib import Path

from dgl import DGLHeteroGraph
from tqdm import tqdm
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import Converter
from grid2op.Action import ActionSpace
from grid2op.Observation import BaseObservation
from grid2op.Environment import BaseEnv
import torch as th
import torch.nn as nn
from torch import Tensor
from torch.utils.tensorboard.writer import SummaryWriter
import dgl
import numpy as np

from GNN.dueling_gcn import DuelingGCN
from agent.replay_buffer import ReplayMemory, Transition

from typing import Tuple, List, Optional


class DoubleDuelingGCNAgent(AgentWithConverter):
    """
    Double Dueling Graph Convolutional Neural (GCN) N2048etwork Agent.
    GCN is used to embed the graph.
    In the Dueling framework the Neural Network predicts:
    - the Value function for the current observation;
    - the Advantage values for each action over the current observation.

    The Q values are then computed by aggregating Value function and Advantage values.
    See :class:`DuelingGCN` for more details.

    In a double GCN two equal models are used:
    - the Q Network is updated at each timestep;
    - the Target Network is updated every :attribute:`replace` steps to avoid bias issues.


    Attributes
    ----------
    q_network: :class:`nn.Module`
        Q Network Model updated at every timestep.

    target_network: :class:`nn.Module`
        Target Network model updated every :replace:`timesteps`

    action_space_converter: :class:`Converter`
        Conversion class to map the action space to a suitable representation for the models

    learning_frequency: `int`
        Number of timesteps between one learning step and another

    replace: ``int``
        Number of steps after which the target network is updated (e.g. the parameters copied from the q_network)

    gamma: ``float``
        MDP charachteristic (not an agent parameter) which defines how much future steps influence current reward

    max_epsilon: ``float``
        Starting value for the decaying exploration parameter

    min_epsilon: ``float``
        Minimum value of the decaying exploration

    epsilon_decay: ``int``
        Exploration decay step, this is the parameter for the exponential decay equation

    learning_rate: ``float``
        Learning rate of the optimizer (Adam automatically handles learning rate decay)

    delta: ``float``
        delta parameter in Huber Loss for tuning between L1 and L2 loss

    alpha: ``float``
        controls how much prioritization is used between 0 and 1

    max_beta: ``float``
        Starting value for the beta parameter, controls bias correction the higher the more correction
        in the Prioritized Experience Replay Buffer

    min_beta: ``float``
        Minimum value for the beta bias correction parameter

    beta_decay: ``int``
        Decay rate for exponential beta decay

    batch_size: ``int``
        Batch size, default value is length of 1 day in grid2op. 24 hours times 60 minutes
        divided by 5 because timestamps are taken every 5 minutes

    tensorboard_log_dir: ``str``
        Directory where to save tensorboard logs. To disable it set it to None.

    training: ``bool``
        If training epsilon-greedy will be used to favour exploration else advantage method will
        be used to select actions
    """

    def __init__(
        self,
        action_space: ActionSpace,
        q_network: DuelingGCN,
        target_network: DuelingGCN,
        action_space_converter: Converter,
        name: str = "gcn_agent",
        kwargs_converter: dict = {
            "all_actions": None,
            "set_line_status": False,
            "change_bus_vect": True,
            "set_topo_vect": False,
        },
        learning_frequency: int = 4,
        replace: int = 500,
        gamma: float = 0.99,
        max_epsilon: float = 1.0,
        min_epsilon: float = 0.01,
        epsilon_decay: int = 200,
        learning_rate: float = 1e-2,
        delta: float = 1.0,
        alpha: float = 0.5,
        max_beta: float = 1.0,
        min_beta: float = 0.0,
        beta_decay: int = 200,
        batch_size: int = int(24 * 60 / 5),  # 1 day in grid2Op environment
        tensorboard_log_dir: str = "./runs/",
        log_dir: str = ".",
        training: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__(action_space, action_space_converter, **kwargs_converter)

        self.name = name
        self.action_space_converter = action_space_converter
        # Initialize Torch device
        if device is None:
            self.device: th.device = th.device(
                "cuda:0" if th.cuda.is_available() else "cpu"
            )
        else:
            self.device: th.device = th.device("cpu")

        print("Selected device: " + str(self.device))

        if str(self.device) == "cpu":
            print("Number of cpu threads: {}".format(th.get_num_threads()))

        # Name Agent for Grid2Op compatibility
        self.name = "gcn_agent"

        # Initialize deep networks
        self.q_network: DuelingGCN = q_network
        self.target_network: DuelingGCN = target_network
        q_network.to(self.device)
        target_network.to(self.device)

        print(self.q_network)
        DuelingGCN.count_parameters(self.q_network)

        # Dueling
        self.replace: int = replace

        # Reporting
        self.trainsteps: int = 0
        self.episodes: int = 0
        self.alive_steps: int = 0
        self.alive_incremental_mean: float = 0
        self.learning_steps: int = 0
        self.reward_incremental_mean: float = 0
        self.cumulative_reward: float = 0

        # Problem charachteristics
        self.gamma: float = gamma

        # Agent Parameters
        self.max_epsilon: float = max_epsilon
        self.min_epsilon: float = min_epsilon
        self.epsilon_decay: int = epsilon_decay
        self.learning_frequency: int = learning_frequency

        # Optimizer
        self.learning_rate = learning_rate
        self.optimizer: th.optim.Optimizer = th.optim.Adam(
            self.q_network.parameters(), lr=learning_rate
        )

        # Action Converter
        self.action_space_converter = action_space_converter
        action_space_converter.seed(0)  # for reproducibility
        action_space_converter.init_converter(**kwargs_converter)

        # Replay Buffer
        self.memory: ReplayMemory = ReplayMemory(int(1e5), alpha)
        self.max_beta: float = max_beta
        self.min_beta: float = min_beta
        self.beta_decay: int = beta_decay

        # Batch Size
        self.batch_size: int = batch_size

        # Logging
        ## Model Checkpoints
        self.log_dir: str = log_dir
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.log_file: str = str(Path(self.log_dir, name + ".pt"))

        # Huber Loss initialization with delta
        self.loss_func: nn.HuberLoss = nn.HuberLoss(delta=delta)

        # Training or Evaluation
        self.training: bool = training
        if training:
            print(
                "\n\nCare training defaults to True, exploration is included in action selection\n\n"
            )
            ## Tensorboard
            # Care SummaryWriter does not support pickling
            # During evaluation pickling is needed for multiprocessing thus this is disabled
            # during evaluation
            Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
            self.writer: Optional[SummaryWriter]
            if tensorboard_log_dir is not None:
                self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
            else:
                self.writer = None

    def compute_loss(
        self, transitions_batch: Transition, sampling_weights: th.Tensor
    ) -> tuple[Tensor, Tensor, DGLHeteroGraph, DGLHeteroGraph]:
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

        Return
        ------
        loss: :class:`th.Tensor`
            Current loss computed as Huber Loss with sampling weights to q_values and td_errors
        td_errors: :class:`th.Tensor`
            Temporal difference errors computed with the Double Q Learning approach, returned to be eventually used for
            priorities inside the prioritized experience replay buffer
        """

        # Unwrap batch
        # Get observation start and end

        observation_batch = self.batch_observations(transitions_batch.observation)
        next_observation_batch = self.batch_observations(
            transitions_batch.next_observation
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
        td_errors: Tensor = rewards + self.gamma * target_q_values.gather(
            1, best_actions
        ).to(self.device)

        # deltas = weights (q_values - td_errors)
        # to keep interfaces general we distribute weights
        loss: Tensor = self.loss_func(
            q_values * sampling_weights, td_errors * sampling_weights
        ).to(self.device)

        return loss, td_errors, observation_batch, next_observation_batch

    def exponential_decay(self, max_val: float, min_val: float, decay: int) -> float:
        """
        Exponential decay schedule, value evolves with trainsteps stored in the agent's state

        Parameters
        ----------
        max_val: ``float``
            upperbound of the returned value, starting point of the decay
        min_val: ``float``
            lowebound of the returned value, ending point of the decay
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
        reward: float,
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
                self.max_epsilon, self.min_epsilon, self.epsilon_decay
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

        return self.to_dgl(observation)

    def save_to_tensorboard(
        self,
        loss: float,
        reward: float,
        observation_batch: th.Tensor,
        next_observation_batch: th.Tensor,
    ) -> None:
        """
        Save to tensorboard:
        - loss
        - reward
        - beta
        - epsilon

        Parameters
        ----------
        loss: ``float``
            current agent loss
        reward: ``float``
            last reward obtained by the agent as a mean over the batch
        """
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
                self.max_epsilon, self.min_epsilon, self.epsilon_decay
            ),
            self.trainsteps,
        )
        self.writer.add_scalar(
            "Beta/train",
            self.exponential_decay(self.max_beta, self.min_beta, self.beta_decay),
            self.trainsteps,
        )

    def learn(self) -> None:
        """
        Learning step for the agent.
        Learing starts after we have at least 'batch' transitions in memory.
        Every 'replace' steps update the target network.
        Experiences are sampled through prioritized experience replay dependent on exponentially decaying beta.
        Loss is computed through Double Dueling Q learning.

        """
        if len(self.memory) < self.batch_size:
            return
        if self.trainsteps % self.replace == 0:
            self.target_network.parameters = self.q_network.parameters

        # Sample from Replay Memory and unpack
        idxs, transitions, sampling_weights = self.memory.sample(
            self.batch_size,
            self.exponential_decay(self.max_beta, self.min_beta, self.beta_decay),
        )
        transitions = Transition(*zip(*transitions))

        loss, td_error, observation_batch, next_observation_batch = self.compute_loss(
            transitions, th.Tensor(sampling_weights)
        )

        # Backward propagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities for sampling
        self.memory.update_priorities(idxs, td_error.abs().detach().numpy().flatten())

        # Save current information to Tensorboard and Checkpoint optimizer and models
        self.save(
            float(loss.mean().item()),
            transitions.reward,
            observation_batch,
            next_observation_batch,
        )

    def save(
        self,
        loss: float,
        rewards: List[float],
        observation_batch: th.Tensor,
        next_observation_batch: th.Tensor,
    ) -> None:
        """
        Save a checkpoint of networks and optimizer.
        Save all the hyperparameters.
        Save Reward, Loss, Epsilon and Beta to Tensorboard.

        Parameters
        ----------
        loss: ``float``
            current agent loss

        rewards: ``List[float]``
            rewards obtained from the last batch
        """
        checkpoint = {
            "optimizer_state": self.optimizer.state_dict(),
            "trainsteps": self.trainsteps,
            "episodes": self.episodes,
            "name": self.name,
            "agent_hyperparameters": {
                "alpha": self.memory.alpha,
                "batch_size": self.batch_size,
                "memory_size": self.memory.buffer_length,
                "gamma": self.gamma,
                "replace_rate": self.replace,
                "max_epsilon": self.max_epsilon,
                "min_epsilon": self.min_epsilon,
                "epsilon_decay": self.epsilon_decay,
                "max_beta": self.max_beta,
                "min_beta": self.min_beta,
                "beta_decay": self.beta_decay,
                "learning_rate": self.learning_rate,
            },
        }
        self.q_network.save()
        self.target_network.save()
        th.save(checkpoint, self.log_file)
        self.save_to_tensorboard(
            loss, np.mean(rewards), observation_batch, next_observation_batch
        )

    def load(self, load_dir: str, networks_loaded: bool = True) -> None:
        """
        Load model from checkpoint

        Parameters
        ----------

        load_dir: ``str``
            Directory of the checkpoint
        networks_loaded: ``bool``
            set it to False if network parameters are included in agent's checkpoint
        """

        checkpoint = th.load(load_dir)

        if not networks_loaded:
            self.q_network.load_state_dict(checkpoint["q_network_state"])
            self.target_network.load_state_dict(checkpoint["target_network_state"])

        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.episodes = checkpoint["episodes"]
        self.checkpoint = checkpoint["trainsteps"]
        self.alpha = checkpoint["agent_hyperparameters"]["alpha"]
        self.batch_size = checkpoint["agent_hyperparameters"]["batch_size"]
        self.gamma = checkpoint["agent_hyperparameters"]["gamma"]
        self.replace_rate = checkpoint["agent_hyperparameters"]["replace_rate"]
        self.max_epsilon = checkpoint["agent_hyperparameters"]["max_epsilon"]
        self.min_epsilon = checkpoint["agent_hyperparameters"]["min_epsilon"]
        self.epsilon_decay = checkpoint["agent_hyperparameters"]["epsilon_decay"]
        self.max_beta = checkpoint["agent_hyperparameters"]["max_beta"]
        self.min_beta = checkpoint["agent_hyperparameters"]["min_beta"]
        self.beta_decay = checkpoint["agent_hyperparameters"]["beta_decay"]
        self.learning_rate = checkpoint["agent_hyperparameters"]["learning_rate"]

        print("Agent Succesfully Loaded!")

    def batch_observations(
        self, observations: Tuple[BaseObservation]
    ) -> dgl.DGLHeteroGraph:
        """
        Convert a list (or tuple) of observations to a Deep Graph Library graph batch.
        A graph batch is represented as a normal graph with nodes and edges added (together with features).

        Parameters
        ----------
        observations: ``Tuple[BaseObservation]``
            tuple of BaseObservation usually stored inside of a :class:`Transition`

        Return
        ------
        graph_batch: :class:`dgl.DGLHeteroGraph`
            a batch of graphs represented as a single augmented graph for Deep Graph Library compatibility
        """
        graphs = []
        for o in observations:
            graph = self.convert_obs(o)
            graphs.append(graph)
        graph_batch = dgl.batch(graphs)
        return graph_batch

    @staticmethod
    def to_dgl(obs: BaseObservation) -> dgl.DGLHeteroGraph:
        """
        convert a :class:BaseObservation to a :class:`dgl.DGLHeteroGraph`.

        Parameters
        ----------
        obs: :class:`BaseObservation`
            BaseObservation taken from a grid2Op environment

        Return
        ------
        dgl_obs: :class:`dgl.DGLHeteroGraph`
            graph compatible with the Deep Graph Library
        """

        # Convert Grid2op graph to a directed (for compatibility reasons) networkx graph
        net = obs.as_networkx()
        net = net.to_directed()  # Typing error from networkx, ignore it

        # Convert from networkx to dgl graph
        dgl_net = dgl.from_networkx(
            net,
            node_attrs=["p", "q", "v", "cooldown"],
            edge_attrs=[
                "rho",
                "cooldown",
                "status",
                "thermal_limit",
                "timestep_overflow",
                "p_or",
                "p_ex",
                "q_or",
                "q_ex",
                "a_or",
                "a_ex",
            ],
        )
        return dgl_net

    def step(
        self,
        observation: BaseObservation,
        action: int,
        reward: float,
        next_observation: BaseObservation,
        done: bool,
    ) -> None:
        """
        Updates the agent's state based on feedback received from the environment.

        Parameters:
        -----------
        observation: :class:`BaseObservation`
            previous observation from the environment
        action: ``int``
            the action taken by the agent in the previous state.
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

            # every so often the agent should learn from experiences
            if self.trainsteps % self.learning_frequency == 0:
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
