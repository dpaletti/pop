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
from itertools import repeat

from node_agents.ray_shallow_gcn_agent import RayShallowGCNAgent


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
        manager_loss: str = "mean",
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
            manager_loss=manager_loss,
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
            if len(action_space) > 1
            else RayShallowGCNAgent.remote(
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
                node_features=self.node_features
                + 2,  # Node Features + Action + Global Node Id
                edge_features=self.edge_features,
                architecture=self.architecture["manager"],
                name="manager_" + str(idx) + "_" + name,
                training=self.training,
            )
            for idx, _ in enumerate(self.communities)
        ]

        self.head_manager = HeadManager(
            node_features=ray.get(self.managers[0].get_embedding_dimension.remote())
            * 2,  # Manager Embedding + Action (padded)
            architecture=self.architecture["head_manager"],
            name="head_manager_" + "_" + name,
            log_dir=self.checkpoint_dir,
            training=self.training,
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
        self.encoded_actions, current_epsilons = zip(
            *ray.get(
                [
                    agent.take_action.remote(transformed_observation=observation)
                    for observation, agent in zip(factored_observation, self.agents)
                ]
            )
        )
        return (
            [
                converter.all_actions[encoded_action]
                for encoded_action, converter in zip(
                    self.encoded_actions, self.agent_converters
                )
            ],
            self.encoded_actions,
            current_epsilons,
        )

    def get_manager_actions(self, subgraphs: List[dgl.DGLHeteroGraph]):
        return zip(
            *ray.get(
                [
                    manager.forward.remote(g=subgraph)
                    for manager, subgraph in zip(self.managers, subgraphs)
                ]
            )
        )

    def get_action(self, summarized_graph: dgl.DGLHeteroGraph):
        return self.head_manager(summarized_graph)

    def step_agents(self, next_observation, reward, done):
        losses = ray.get(
            [
                agent.step.remote(
                    observation=agent_observation,
                    action=agent_action,
                    reward=reward,
                    next_observation=agent_next_observation,
                    done=done,
                    stop_decay=False
                    if (
                        idx == self.current_chosen_node and self.epsilon_beta_scheduling
                    )
                    else True,
                )
                for (
                    (idx, agent),
                    agent_action,
                    agent_observation,
                    agent_next_observation,
                    _,
                ) in zip(
                    enumerate(self.agents),
                    self.encoded_actions,
                    self.factored_observation,
                    *factor_observation(next_observation, self.device),
                )
            ]
        )
        return losses, list(map(lambda x: not (x is None), losses))

    def teach_managers(self, reward):
        # TODO: see https://arxiv.org/abs/1506.05254
        # TODO: see score function in https://pytorch.org/docs/stable/distributions.html
        ray.get(
            [
                manager.learn.remote(reward=r)
                for manager, r in zip(self.managers, repeat(reward, len(self.managers)))
            ]
        )

    def learn(self, reward: float, losses: List[th.Tensor]):
        if not self.fixed_communities:
            raise Exception("In Learn() we assume fixed communities")

        self.teach_managers(reward)

        head_manager_loss: th.Tensor = (
            -self.head_manager.node_choice.attention_distribution.log_prob(
                th.tensor(self.head_manager.current_best_node)
            )
            * reward
        )

        self.head_manager_optimizer.zero_grad()
        head_manager_loss.backward()
        th.nn.utils.clip_grad_norm_(
            self.head_manager.parameters(), self.head_manager.architecture["max_clip"]
        )
        self.head_manager_optimizer.step()

        # Summary Writer is supposed to not slow down training
        self.save_to_tensorboard(head_manager_loss.item(), reward)

    @staticmethod
    def load(
        checkpoint_file: str,
        training: bool,
        device: str,
        tensorboard_dir: Optional[str],
        checkpoint_dir: Optional[str],
        reset_decay: bool = True,
        new_name: Optional[str] = None,
        new_env: Optional[BaseEnv] = None,
        n_jobs: int = 1,
    ):
        print("Loading Model")
        checkpoint = th.load(checkpoint_file)
        rayDPOP = RayDPOP(
            env=checkpoint["env"],
            name=checkpoint["name"] if new_name is None else new_name,
            architecture=checkpoint["architecture"],
            training=training,
            tensorboard_dir=tensorboard_dir,
            checkpoint_dir=checkpoint_dir,
            seed=checkpoint["seed"],
            device=device,
            n_jobs=n_jobs,
        )

        for agent, agent_state in zip(rayDPOP.agents, checkpoint["agents"].values()):
            agent.load_state.remote(**agent_state, reset_decay=reset_decay)

        for manager, manager_state in zip(
            rayDPOP.managers, checkpoint["managers"].values()
        ):
            manager.load_state.remote(**manager_state)

        rayDPOP.head_manager.load_state_dict(checkpoint["head_manager"]["state"])
        rayDPOP.head_manager_optimizer.load_state_dict(
            checkpoint["head_manager"]["optimizer_state"]
        )

        rayDPOP.managers_learning_steps = checkpoint["learning_steps"]
        rayDPOP.alive_steps = checkpoint["alive_steps"]
        rayDPOP.trainsteps = checkpoint["trainsteps"]
        rayDPOP.episodes = checkpoint["episodes"]

        print("Model Loaded")

        return rayDPOP

    def save(self):
        agents_state = ray.get([agent.get_state.remote() for agent in self.agents])
        agents_name = ray.get([agent.get_name.remote() for agent in self.agents])
        agents_dict = {
            name: {
                kword: state
                for state, kword in zip(
                    state,
                    [
                        "optimizer_state",
                        "q_network_state",
                        "target_network_state",
                        "losses",
                        "actions",
                        "decay_steps",
                        "alive_steps",
                        "trainsteps",
                        "learning_steps",
                    ],
                )
            }
            for name, state in zip(agents_name, agents_state)
        }

        head_manager_dict = {
            "state": self.head_manager.state_dict(),
            "optimizer_state": self.head_manager_optimizer.state_dict(),
            "name": self.head_manager.name,
        }

        managers_state = ray.get(
            [manager.get_state.remote() for manager in self.managers]
        )
        managers_name = ray.get(
            [manager.get_name.remote() for manager in self.managers]
        )

        managers_dict = {
            name: {
                kword: state
                for state, kword in zip(
                    state,
                    [
                        "embedding_state_dict",
                        "node_attention_state_dict",
                        "optimizer_state_dict",
                        "actions",
                        "losses",
                    ],
                )
            }
            for name, state in zip(managers_name, managers_state)
        }

        checkpoint = {
            "name": self.name,
            "env": self.env,
            "architecture": self.architecture,
            "seed": self.seed,
            "agents": agents_dict,
            "managers": managers_dict,
            "head_manager": head_manager_dict,
            "learning_steps": self.managers_learning_steps,
            "trainsteps": self.trainsteps,
            "alive_steps": self.alive_steps,
            "episodes": self.episodes,
        }
        th.save(
            checkpoint,
            self.checkpoint_file,
        )
