from pathlib import Path
from typing import List, Optional, Tuple, Union
import torch as th


import dgl
import networkx as nx
import grid2op
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation

from managers.async_community_manager import AsyncCommunityManager
from managers.head_manager import HeadManager
from multiagent_system.base_pop import BasePOP
from node_agents.async_gcn_agent import AsyncGCNAgent
from pop.multiagent_system.task import Task, TaskType
from pop.multiagent_system.space_factorization import (
    factor_observation,
)
from pop.node_agents.utilities import from_networkx_to_dgl
from torch.multiprocessing import Manager, Queue


# TODO: tracking communities in dynamic graphs
# TODO: https://www.researchgate.net/publication/221273637_Tracking_the_Evolution_of_Communities_in_Dynamic_Social_Networks


class AsyncDPOP(BasePOP):
    queue_manager: Manager = Manager()

    def __init__(
        self,
        env: BaseEnv,
        name: str,
        architecture: Union[str, dict],
        training: bool,
        tensorboard_dir: Optional[str],
        checkpoint_dir: Optional[str],
        seed: int,
        device: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ):
        super(AsyncDPOP, self).__init__(
            env=env,
            name=name,
            architecture=architecture,
            training=training,
            tensorboard_dir=tensorboard_dir,
            checkpoint_dir=checkpoint_dir,
            seed=seed,
            device=device,
            n_jobs=n_jobs,
        )

        # Multiprocessing
        self.result_queue: Queue = self.queue_manager.Queue()

        self.encoded_actions: List[int] = []
        self._agents: List[AsyncGCNAgent] = [
            AsyncGCNAgent(
                agent_actions=len(action_space),
                architecture=self.architecture["agent"],
                node_features=self.node_features,
                edge_features=self.edge_features,
                name="agent_" + str(idx) + "_" + name,
                training=training,
                device=device,
                task_queue=self.queue_manager.Queue(),
                result_queue=self.result_queue,
            )
            for idx, action_space in enumerate(self.action_spaces)
        ]

        # Start agent process
        for agent in self.agents:
            agent.start()

        # Managers Initialization
        self._managers: List[AsyncCommunityManager] = [
            AsyncCommunityManager(
                node_features=self.node_features + 1,  # Node Features + Action
                edge_features=self.edge_features,
                architecture=self.architecture["manager"],
                name="manager_" + str(idx) + "_" + name,
                task_queue=self.queue_manager.Queue(),
                result_queue=self.result_queue,
            ).to(device)
            for idx, _ in enumerate(self.communities)
        ]

        for manager in self.managers:
            manager.start()

        self.head_manager = HeadManager(
            node_features=self.managers[0].get_embedding_dimension()
            * 2,  # Manager Embedding + Action (padded)
            architecture=self.architecture["head_manager"],
            name="head_manager_" + "_" + name,
            log_dir=self.checkpoint_dir,
        ).to(device)

        self.head_manager_optimizer: th.optim.Optimizer = th.optim.Adam(
            self.head_manager.parameters(),
            lr=self.architecture["head_manager"]["learning_rate"],
        )

    @property
    def agents(self):
        return self._agents

    @property
    def managers(self):
        return self._managers

    def get_agent_actions(self, factored_observation):
        for observation, agent in zip(factored_observation, self.agents):
            agent.task_queue.put(
                Task(TaskType.ACT, transformed_observation=observation)
            )

        self.encoded_actions = self.harvest_agent_results()

        return [
            converter.all_actions[encoded_action]
            for encoded_action, converter in zip(
                self.encoded_actions, self.agent_converters
            )
        ]

    def harvest_agent_results(self):
        result_list = []
        for _ in self.agents:
            result_list.append(self.result_queue.get())

        result_list = [
            (int(agent.split("_")[1]), result) for (agent, result) in result_list
        ]
        result_list = sorted(result_list, key=lambda tup: tup[0])

        return [result for (_, result) in result_list]

    def harvest_manager_results(self):
        result_list = []
        for _ in self.managers:
            result_list.append(self.result_queue.get())

        result_list = [
            (int(manager.split("_")[1]), result) for (manager, result) in result_list
        ]
        result_list = sorted(result_list, key=lambda tup: tup[0])

        return [result for (_, result) in result_list]

    def get_manager_actions(self, subgraphs: List[dgl.DGLHeteroGraph]):
        for mangager, subgraph in zip(self.managers, subgraphs):
            mangager.task_queue.put(Task(TaskType.CHOOSE_ACTION, g=subgraph))

        return zip(*self.harvest_manager_results())

    def teach_managers(self, manager_losses):
        for manager, manager_loss in zip(self.managers, manager_losses):
            manager.task_queue.put(Task(TaskType.LEARN, loss=manager_loss))
        self.harvest_manager_results()

    def step_agents(
        self, next_observation, reward, done
    ) -> Tuple[List[th.Tensor], List[bool]]:
        for (agent, agent_action, agent_observation, agent_next_observation, _,) in zip(
            self.agents,
            self.encoded_actions,
            self.factored_observation,
            *factor_observation(next_observation, self.device),
        ):
            agent.task_queue.put(
                Task(
                    TaskType.STEP,
                    observation=agent_observation,
                    action=agent_action,
                    reward=reward,
                    next_observation=agent_next_observation,
                    done=done,
                )
            )

        losses = self.harvest_agent_results()

        return losses, list(map(lambda x: not (x is None), losses))

    def save(self):
        for agent in self.agents:
            agent.task_queue.put(Task(TaskType.SAVE))
        agents_state = dict(self.harvest_agent_results())

        for manager in self.managers:
            manager.task_queue.put(Task(TaskType.SAVE))
        managers_state = dict(self.harvest_manager_results())

        checkpoint = {
            "agents_state": agents_state,
            "managers_state": managers_state,
            "head_manager_state": self.head_manager.state_dict(),
            "head_manager_optimizer_state": self.head_manager_optimizer.state_dict(),
            "trainsteps": self.trainsteps,
            "episodes": self.episodes,
            "name": self.name,
            "env_name": self.env.env_name,
            "architecture": self.architecture,
            "seed": self.seed,
        }
        th.save(checkpoint, self.checkpoint_dir)

    @staticmethod
    def load(
        checkpoint_file: str,
        training: bool,
        device: str,
        tensorboard_dir: Optional[str] = None,
    ):
        checkpoint = th.load(checkpoint_file)

        dpop = AsyncDPOP(
            env=grid2op.make(checkpoint["env_name"]),
            architecture=checkpoint["architecture"],
            name=checkpoint["name"],
            training=training,
            tensorboard_dir=tensorboard_dir,
            checkpoint_dir=Path(checkpoint_file).parents[0],
            seed=checkpoint["seed"],
            device=device,
        )

        dpop.trainsteps = checkpoint["trainsteps"]
        dpop.episodes = checkpoint["episodes"]

        for idx, agent in enumerate(dpop.agents):
            agent.task_queue.put(
                Task(
                    TaskType.LOAD,
                    optimizer_state=checkpoint["agents_state"][
                        "agent_" + str(idx) + "_" + dpop.name
                    ]["optimizer_state"],
                    q_network_state=checkpoint["agents_state"][
                        "agent_" + str(idx) + "_" + dpop.name
                    ]["q_network_state"],
                    target_network_state=checkpoint["agents_state"][
                        "agent_" + str(idx) + "_" + dpop.name
                    ]["target_network_state"],
                    losses=checkpoint["agents_state"][
                        "agent_" + str(idx) + "_" + dpop.name
                    ]["losses"],
                    actions=checkpoint["agents_state"][
                        "agent_" + str(idx) + "_" + dpop.name
                    ]["actions"],
                )
            )
        dpop.harvest_agent_results()

        for idx, manager in enumerate(dpop.managers):
            manager.task_queue.put(
                Task(
                    TaskType.LOAD,
                    state_dict=checkpoint["managers_state"][
                        "manager_" + str(idx) + "_" + dpop.name
                    ]["state"],
                    optimizer_state_dict=checkpoint["managers_state"][
                        "manager_" + str(idx) + "_" + dpop.name
                    ]["optimizer_state"],
                    action=checkpoint["managers_state"][
                        "manager_" + str(idx) + "_" + dpop.name
                    ]["chosen_actions"],
                    losses=checkpoint["managers_state"][
                        "manager_" + str(idx) + "_" + dpop.name
                    ]["losses"],
                )
            )
        dpop.harvest_manager_results()
