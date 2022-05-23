from typing import List, Union, Optional

import dgl
from grid2op.Environment import BaseEnv
import ray

from managers.head_manager import HeadManager
from managers.ray_community_manager import RayCommunityManager
from multiagent_system.base_pop import BasePOP
from multiagent_system.space_factorization import factor_observation
from node_agents.ray_gcn_agent import RayGCNAgent
import torch as th


class RayDPOP(BasePOP):
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
        super(RayDPOP, self).__init__(
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

        # Agents Initialization
        self.encoded_actions: List[int] = []
        self._agents: List[RayGCNAgent] = [
            RayGCNAgent.remote(
                agent_actions=len(action_space),
                architecture=self.architecture["agent"],
                node_features=self.node_features,
                edge_features=self.edge_features,
                name="agent_" + str(idx) + "_" + name,
                training=training,
                device=device,
            )
            for idx, action_space in enumerate(self.action_spaces)
        ]

        # Managers Initializations
        self._managers: List[RayCommunityManager] = [
            RayCommunityManager.remote(
                node_features=self.node_features + 1,  # Node Features + Action
                edge_features=self.edge_features,
                architecture=self.architecture["manager"],
                name="manager_" + str(idx) + "_" + name,
            )
            for idx, _ in enumerate(self.communities)
        ]

        self.head_manager = HeadManager(
            node_features=ray.get(self.managers[0].get_embedding_dimension.remote())
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
        self.encoded_actions = ray.get(
            [
                agent.take_action.remote(transformed_observation=observation)
                for observation, agent in zip(factored_observation, self.agents)
            ]
        )
        return [
            converter.all_actions[encoded_action]
            for encoded_action, converter in zip(
                self.encoded_actions, self.agent_converters
            )
        ]

    def get_manager_actions(self, subgraphs: List[dgl.DGLHeteroGraph]):
        return zip(
            *ray.get(
                [
                    manager.forward.remote(g=subgraph)
                    for manager, subgraph in zip(self.managers, subgraphs)
                ]
            )
        )

    def step_agents(self, next_observation, reward, done):
        losses = ray.get(
            [
                agent.step.remote(
                    observation=agent_observation,
                    action=agent_action,
                    reward=reward,
                    next_observation=agent_next_observation,
                    done=done,
                )
                for (
                    agent,
                    agent_action,
                    agent_observation,
                    agent_next_observation,
                    _,
                ) in zip(
                    self.agents,
                    self.encoded_actions,
                    self.factored_observation,
                    *factor_observation(
                        next_observation, self.edge_features, self.device
                    ),
                )
            ]
        )
        return losses, list(map(lambda x: not (x is None), losses))

    def teach_managers(self, manager_losses):
        ray.get(
            [
                manager.learn.remote(loss=loss)
                for manager, loss in zip(self.managers, manager_losses)
            ]
        )

    @staticmethod
    def load(
        checkpoint_file: str,
        training: bool,
        device: str,
        tensorboard_dir: Optional[str] = None,
    ):
        raise Exception("TO IMPLEMENT")

    def save(self):
        raise Exception("TO IMPLEMENT")
