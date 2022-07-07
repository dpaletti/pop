from typing import List, Union, Optional, Dict, Set, Tuple

import dgl
import networkx as nx
from grid2op.Environment import BaseEnv
import ray

from configs.architecture import Architecture
from multiagent_system.base_pop import BasePOP
from multiagent_system.space_factorization import (
    factor_observation,
    split_graph_into_communities,
)
from agents.ray_gcn_agent import RayGCNAgent
import torch as th
from itertools import repeat

from agents.ray_shallow_gcn_agent import RayShallowGCNAgent


# TODO: we can try not to factor the observation space but only the action space
# TODO: the idea here is that each node sees all the graph but can act only on a subset
# TODO: the manager takes actions only from a community (or more)
# TODO: but still sees all the graph
# TODO: this may greatly stabilise the training (removing partial observability) while
# TODO: still retaining the advantages of sensible action space factorization
# TODO: we can achieve this with radius < 0 and a switch in the observation factorization method
class RayDPOP(BasePOP):
    def __init__(
        self,
        env: BaseEnv,
        name: str,
        architecture: Architecture,
        training: bool,
        tensorboard_dir: Optional[str],
        checkpoint_dir: Optional[str],
        seed: int,
        device: Optional[str] = None,
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
        )

        # Agents Initialization
        self.encoded_actions: List[int] = []
        self._agents: List[RayGCNAgent] = [
            RayGCNAgent.remote(
                agent_actions=len(action_space),
                architecture=self.architecture.agent,
                node_features=self.node_features,
                edge_features=self.edge_features,
                name="agent_" + str(idx) + "_" + name,
                training=training,
                device=device,
            )
            if len(action_space) > 1
            else RayShallowGCNAgent.remote(
                agent_actions=len(action_space),
                architecture=self.architecture.agent,
                node_features=self.node_features,
                edge_features=self.edge_features,
                name="agent_" + str(idx) + "_" + name,
                training=training,
                device=device,
            )
            for idx, action_space in enumerate(self.action_spaces)
        ]

        # Managers Initialization
        self._managers: Dict[Tuple[int, ...], RayGCNAgent] = {
            community: RayGCNAgent.remote(
                agent_actions=self.graph.nodes,
                node_features=self.node_features
                + 2,  # Node Features + Action + Global Node Id
                edge_features=self.edge_features,
                architecture=self.architecture.manager,
                name="manager_" + str(idx) + "_" + self.name,
                training=self.training,
                device=device,
            )
            for idx, community in enumerate(self.communities)
        }

        # Head Manager Initialization
        self.head_manager: RayGCNAgent = RayGCNAgent(
            agent_actions=self.graph.nodes,
            node_features=ray.get(
                list(self.managers.values())[
                    0
                ].q_network.embedding.get_embedding_dimension()
            ),
            architecture=self.architecture.head_manager,
            name="head_manager_" + name,
            log_dir=None,
            training=self.training,
            device=device,
        )

        self.head_manager_optimizer: th.optim.Optimizer = th.optim.Adam(
            self.head_manager.parameters(),
            lr=self.architecture.head_manager.learning_rate,
        )

    @property
    def agents(self) -> List[RayGCNAgent]:
        return self._agents

    @property
    def managers(self) -> Dict[Tuple[int, ...], RayGCNAgent]:
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

    def get_manager_actions(
        self, subgraphs: Dict[Tuple[int, ...], dgl.DGLHeteroGraph]
    ) -> Tuple[List[int], List[float]]:
        chosen_nodes: List[int]
        epsilons: List[float]
        chosen_nodes, epsilons = zip(
            *ray.get(
                [
                    self.managers[community].take_action(
                        subgraphs[community], mask=community
                    )
                    for community in self.communities
                ]
            )
        )

        return (
            chosen_nodes,
            epsilons,
        )

    def get_action(self, graph: dgl.DGLHeteroGraph):
        return self.head_manager.take_action(graph, mask=graph.nodes)

    def step_agents(self, next_observation, reward, done):
        losses, q_network_states, target_network_states = zip(
            *ray.get(
                [
                    agent.step.remote(
                        observation=agent_observation,
                        action=agent_action,
                        reward=reward,
                        next_observation=agent_next_observation,
                        done=done,
                        stop_decay=False
                        if (
                            idx == self.current_chosen_node
                            and self.epsilon_beta_scheduling
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
        )
        return (
            losses,
            q_network_states,
            target_network_states,
            list(map(lambda x: not (x is None), losses)),
        )

    def teach_managers(self, reward):
        return zip(
            *ray.get(
                [
                    manager.learn.remote(reward=r)
                    for manager, r in zip(
                        self.managers, repeat(reward, len(self.managers))
                    )
                ]
            )
        )

    def learn(self, reward: float):
        if not self.fixed_communities:
            raise Exception("In Learn() we assume fixed communities")

        self.writer.add_scalar("POP/Reward", reward, self.train_steps)
        manager_losses, manager_state_dicts = self.teach_managers(reward)

        if not (None in manager_losses):
            for idx, loss in enumerate(manager_losses):
                self.writer.add_scalar(
                    "Manager Loss/Manager " + str(idx),
                    loss,
                    self.managers_learning_steps,
                )

            for idx, state in enumerate(manager_state_dicts):
                for k, v in state.items():
                    if v is not None:
                        self.writer.add_histogram(
                            "Manager " + str(k) + "/Manager " + str(idx),
                            v,
                            self.managers_learning_steps,
                        )

        self.head_manager_mini_batch.append(
            (
                self.head_manager.node_choice.attention_distribution,
                self.head_manager.current_best_node,
                reward,
            )
        )

        if (
            len(self.head_manager_mini_batch)
            >= self.head_manager.architecture["batch_size"]
        ):
            head_manager_losses = [
                -distribution.log_prob(th.tensor(node)) * reward
                for (
                    distribution,
                    node,
                    reward,
                ) in self.head_manager_mini_batch
            ]
            head_manager_loss = head_manager_losses[0]
            for loss in head_manager_losses[1:]:
                head_manager_loss += loss

            head_manager_loss /= len(head_manager_losses)

            self.head_manager_optimizer.zero_grad()
            head_manager_loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.head_manager.parameters(),
                self.head_manager.architecture["max_clip"],
            )
            self.head_manager_optimizer.step()

            self.head_manager_mini_batch = []

            self.writer.add_scalar(
                "Manager Loss/Head Manager",
                head_manager_loss,
                self.managers_learning_steps,
            )
            for k, v in self.head_manager.state_dict().items():
                if v is not None:
                    self.writer.add_histogram(
                        "Manager " + str(k) + "/ Head Manager",
                        v,
                        self.managers_learning_steps,
                    )

            # WARNING: Assuming same mini-batch size for managers and head managers
            self.managers_learning_steps += 1

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
        rayDPOP.train_steps = checkpoint["train_steps"]
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
                        "train_steps",
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
            "train_steps": self.train_steps,
            "alive_steps": self.alive_steps,
            "episodes": self.episodes,
        }
        th.save(
            checkpoint,
            self.checkpoint_file,
        )
