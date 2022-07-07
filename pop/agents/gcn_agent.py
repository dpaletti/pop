from pathlib import Path

from dgl import DGLHeteroGraph
from tqdm import tqdm
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
from grid2op.Action import BaseAction, ActionSpace
from grid2op.Observation import BaseObservation
from grid2op.Environment import BaseEnv
import torch as th
from torch.utils.tensorboard.writer import SummaryWriter
import dgl
import numpy as np

from agents.base_gcn_agent import BaseGCNAgent
from pop.dueling_networks.dueling_net_factory import get_dueling_net

from typing import List, Optional, Union, Tuple

from pop.agents.utilities import to_dgl


class GCNAgent(AgentWithConverter, BaseGCNAgent):
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
        tensorboard_log_dir: Optional[str],
        log_dir,
        device: Optional[str] = None,
    ):
        BaseGCNAgent.__init__(
            self,
            agent_actions=len(agent_actions),
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture,
            name=name,
            training=training,
            device=device,
            log_dir=log_dir,
            tensorboard_dir=tensorboard_log_dir,
        )

        AgentWithConverter.__init__(
            self,
            full_action_space,
            IdToAct,
            all_actions=agent_actions,
            **self.architecture["converter_kwargs"]
        )

        # Action Converter
        self.action_space_converter = IdToAct(full_action_space)
        self.action_space_converter.init_converter(all_actions=agent_actions)
        self.action_space_converter.seed(seed)
        self.seed = seed

        # Logging
        self.log_dir: str = log_dir
        Path(self.log_dir).mkdir(parents=True, exist_ok=False)
        self.log_file: str = str(Path(self.log_dir, name + ".pt"))

        # Training or Evaluation
        self.training: bool = training
        self.tensorboard_log_dir: str = tensorboard_log_dir
        if training:

            # Tensorboard
            self.writer: Optional[SummaryWriter]
            if tensorboard_log_dir is not None:
                Path(tensorboard_log_dir).mkdir(parents=True, exist_ok=False)
                self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
            else:
                self.writer = None

    def convert_obs(self, observation: BaseObservation) -> dgl.DGLHeteroGraph:
        return to_dgl(observation, self.device)

    def my_act(
        self, transformed_observation: DGLHeteroGraph, reward=None, done=False
    ) -> int:
        return super().take_action(transformed_observation)[0]

    def save(
        self,
        loss: float,
        rewards: List[float],
    ) -> None:
        checkpoint = {
            "optimizer_state": self.optimizer.state_dict(),
            "train_steps": self.trainsteps,
            "episodes": self.episodes,
            "name": self.name,
            "architecture": self.architecture,
            "seed": self.seed,
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "tensorboard_log_dir": self.tensorboard_log_dir,
        }
        self.q_network.save()
        self.target_network.save()
        th.save(checkpoint, self.log_file)
        if self.writer is not None:
            self.save_to_tensorboard(loss, np.mean(rewards))

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

    @classmethod
    def load(
        cls,
        load_file: str,
        agent_actions: List[BaseAction],
        full_action_space: ActionSpace,
        device: Union[str, th.device],
        training: bool,
    ):

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

        agent = GCNAgent(
            agent_actions=agent_actions,
            full_action_space=full_action_space,
            node_features=checkpoint["node_features"],
            edge_features=checkpoint["edge_features"],
            architecture=checkpoint["architecture"],
            name=checkpoint["name"],
            seed=checkpoint["seed"],
            training=training,
            tensorboard_log_dir=checkpoint["tensorboard_log_dir"],
            log_dir=Path(load_file).parents[0],
        )

        agent.q_network = q_net
        agent.target_network = target_net

        agent.q_network.load_state_dict(q_net["network_state"])
        agent.target_network.load_state_dict(target_net["network_state"])
        agent.optimizer.load_state_dict(checkpoint["optimizer_state"])

        return agent


def train(
    env: BaseEnv,
    iterations: int,
    agent: GCNAgent,
):
    training_step: int = 0
    obs: BaseObservation = env.reset()
    done = False
    reward = env.reward_range[0]
    total_episodes = len(env.chronics_handler.subpaths)
    with tqdm(total=iterations - training_step) as pbar:
        while training_step < iterations:
            if agent.episodes % total_episodes == 0:
                env.chronics_handler.shuffle()
            if done:
                obs = env.reset()
            encoded_action = agent.my_act(agent.convert_obs(obs))
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
