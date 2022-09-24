import itertools
import time
from abc import abstractmethod
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union

import dgl
import networkx as nx
import numpy as np
import ray
import torch as th
from grid2op.Action import BaseAction
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation
from tqdm import tqdm

from pop.agents.loggable_module import LoggableModule
from pop.agents.manager import Manager
from pop.agents.ray_gcn_agent import RayGCNAgent
from pop.agents.ray_shallow_gcn_agent import RayShallowGCNAgent
from pop.community_detection.community_detector import Community, CommunityDetector
from pop.configs.architecture import Architecture
from pop.multiagent_system.action_detector import ActionDetector
from pop.multiagent_system.fixed_set import FixedSet
from pop.multiagent_system.space_factorization import (
    EncodedAction,
    HashableAction,
    Substation,
    factor_action_space,
    factor_observation,
    split_graph_into_communities,
)
from pop.networks.serializable_module import SerializableModule


class BasePOP(AgentWithConverter, SerializableModule, LoggableModule):
    def __init__(
        self,
        env: BaseEnv,
        name: str,
        architecture: Architecture,
        training: bool,
        seed: int,
        checkpoint_dir: Optional[str] = None,
        tensorboard_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        AgentWithConverter.__init__(self, env.action_space, IdToAct)
        SerializableModule.__init__(self, checkpoint_dir, name)
        LoggableModule.__init__(self, tensorboard_dir)

        self.name = name
        self.seed: int = seed
        self.env = env
        self.node_features: int = architecture.pop.node_features
        self.edge_features: int = architecture.pop.edge_features

        # Converter
        self.converter = IdToAct(env.action_space)
        self.converter.init_converter()
        self.converter.seed(seed)

        # Setting the device
        self.device: th.device
        if device is None:
            self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
        else:
            self.device = th.device(device)

        self.architecture: Architecture = architecture

        # Node and edge features
        self.env = env

        # Training or Evaluation
        self.training = training

        # Logging
        self.train_steps: int = 0
        self.episodes: int = 0
        self.alive_steps: int = 0
        self.community_update_steps: int = (
            0  # Updated until manager initialization is over
        )

        # State needed for step()
        # This state is saved so that after my_act()
        # step() can use pre-computed values
        self.chosen_action: Optional[EncodedAction] = None
        self.community_to_substation: Optional[Dict[Community, EncodedAction]] = None
        self.substation_to_local_action: Optional[
            Dict[Substation, EncodedAction]
        ] = None
        self.sub_graphs: Optional[Dict[Community, dgl.DGLHeteroGraph]] = None
        self.summarized_graph: Optional[dgl.DGLHeteroGraph] = None
        self.factored_observation: Optional[Dict[Substation, dgl.DGLHeteroGraph]] = None
        self.chosen_community: Optional[Community] = None
        self.chosen_node: Optional[int] = None
        self.substation_to_encoded_action: Optional[
            Dict[Substation, EncodedAction]
        ] = None
        self.old_graph: Optional[nx.Graph] = None

        # Agents
        self.action_lookup_table: Optional[Dict[HashableAction, int]] = None
        self.substation_to_action_converter: Optional[Dict[Substation, IdToAct]] = None
        self.substation_to_agent: Optional[
            Dict[Substation, Union[RayGCNAgent, RayShallowGCNAgent]]
        ] = None

        # Agents Initialization
        substation_to_action_space, self.action_lookup_table = factor_action_space(
            env.observation_space, self.converter, self.env.n_sub
        )

        self.substation_to_action_converter = self._get_substation_to_agent_mapping(
            substation_to_action_space
        )

        self.log_action_space_size(agent_converters=self.substation_to_action_converter)

        self.substation_to_agent = {
            sub_id: RayGCNAgent.remote(
                agent_actions=len(action_space),
                architecture=self.architecture.agent,
                node_features=self.node_features,
                edge_features=self.edge_features,
                name="agent_" + str(sub_id) + "_" + self.name,
                training=self.training,
                device=self.device,
            )
            if len(action_space) > 1
            else RayShallowGCNAgent.remote(
                name="agent_" + str(sub_id) + "_" + self.name,
                device=self.device,
            )
            for sub_id, action_space in substation_to_action_space.items()
        }
        # Managers
        self.community_to_manager: Optional[Dict[Community, Manager]] = None
        self.managers_history: Dict[Manager, FixedSet] = {}
        self.manager_initialization_threshold: int = 1

        # Community Detector Initialization
        self.community_detector = CommunityDetector(
            seed,
            enable_power_supply_modularity=architecture.pop.enable_power_supply_modularity,
        )
        self.communities: Optional[List[Community]] = None

        self.action_detector: ActionDetector = ActionDetector(
            loop_length=self.architecture.pop.disabled_action_loops_length,
            penalty_value=self.architecture.pop.repeated_action_penalty,
            repeatable_actions=[0],
        )

    @abstractmethod
    def get_action(self, observation: dgl.DGLHeteroGraph) -> int:
        """
        Method to get System action given manager's action encoded in the observation
        """
        ...

    def _extra_step(
        self,
        action: int,
        reward: float,
        next_sub_graphs: Dict[Community, dgl.DGLHeteroGraph],
        next_substation_to_encoded_action: Dict[Substation, EncodedAction],
        next_graph: nx.Graph,
        done: bool,
        next_communities: List[Community],
        next_community_to_manager: Dict[Community, Manager],
    ):
        """
        Override this to add functionalities to the step() method
        e.g. Children adds a head manager which needs to step
        """
        pass

    def my_act(
        self,
        transformed_observation: Tuple[
            Dict[Substation, Optional[dgl.DGLHeteroGraph]], nx.Graph
        ],
        reward: float,
        done=False,
    ) -> EncodedAction:
        """
        Method implicitly called by act().
        By calling act(observation) you are actually calling my_act(convert_obs(observation))
        This method is inherited from AgentWithConverter from Grid2Op
        """

        graph: nx.Graph
        self.factored_observation, graph = transformed_observation

        self.log_graph(graph, self.train_steps)

        # Update Communities
        # At each episode the community detection algorithm is reset by setting old_graph to None
        self.communities, self.community_to_manager = (
            self._update_communities(graph)
            if self.old_graph is None
            else self._update_communities(self.old_graph, graph)
        )

        self.old_graph = graph.copy()

        # Actions retrieved are encoded with local agent converters
        # need to be remapped to the global converter
        self.substation_to_local_action = self._get_agent_actions(
            self.factored_observation
        )

        # Split graph into communities
        (
            self.sub_graphs,
            self.substation_to_encoded_action,
        ) = self._compute_managers_sub_graphs(
            graph=graph, substation_to_local_action=self.substation_to_local_action
        )

        # Managers chooses the best substation for each community they handle
        self.community_to_substation = self._get_manager_actions(
            self.sub_graphs,
            self.substation_to_encoded_action,
            self.communities,
            self.community_to_manager,
        )

        # Build a summarized graph by mapping communities to supernode
        self.summarized_graph = self._compute_summarized_graph(
            graph=graph,
            sub_graphs=self.sub_graphs,
            substation_to_encoded_action=self.substation_to_encoded_action,
            community_to_substation=self.community_to_substation,
        )

        # The head manager chooses the best action from every community given the summarized graph
        self.chosen_node = self.get_action(self.summarized_graph)
        self.chosen_action = graph.nodes[self.chosen_node]["action"]
        self.chosen_community = next(
            filter(lambda community: self.chosen_node in community, self.communities)
        )

        manager_names: List[str] = ray.get(
            [
                manager.get_name.remote()
                for manager in list(self.community_to_manager.values())
            ]
        )
        community_to_names: Dict[Community, str] = {
            community: "_".join(manager_names[idx].split("_")[0:2])
            for idx, community in enumerate(list(self.community_to_manager.keys()))
        }

        agent_names: List[str] = ray.get(
            [
                agent.get_name.remote()
                for agent in list(self.substation_to_agent.values())
            ]
        )
        substation_to_names: Dict[Substation, str] = {
            substation: "_".join(agent_names[idx].split("_")[0:2])
            for idx, substation in enumerate(list(self.substation_to_agent.keys()))
        }

        # Log to Tensorboard
        self.log_system_behaviour(
            best_action=self.chosen_action,
            best_action_str=str(self.converter.all_actions[self.chosen_action]),
            head_manager_action=self.chosen_node,
            manager_actions={
                community: (action, community_to_names[community])
                for community, action in self.community_to_substation.items()
            },
            agent_actions={
                substation_to_names[sub_id]: action
                for sub_id, action in self.substation_to_local_action.items()
            },
            manager_explorations={
                name: exploration_state
                for name, exploration_state in zip(
                    manager_names,
                    ray.get(
                        [
                            manager.get_exploration_logs.remote()
                            for manager in self.community_to_manager.values()
                        ]
                    ),
                )
            },
            agent_explorations={
                name: exploration_state
                for name, exploration_state in zip(
                    agent_names,
                    ray.get(
                        [
                            agent.get_exploration_logs.remote()
                            for agent in self.substation_to_agent.values()
                        ]
                    ),
                )
            },
            train_steps=self.train_steps,
        )

        if self.action_detector.is_repeated(self.chosen_action):
            self.chosen_action = 0
        return self.chosen_action

    def convert_obs(
        self, observation: BaseObservation
    ) -> Tuple[Dict[Substation, Optional[dgl.DGLHeteroGraph]], nx.Graph]:
        """
        Upon calling act(observation)
        The agent calls my_act(convert_obs(observation))
        This method is thus an implicit observation converter for the agent.
        Inherited from AgentWithConverter (Grid2Op)
        """

        # Observation as a "normal graph"
        observation_graph: nx.Graph = observation.as_networkx()

        # Observation is factored for each agent by taking the ego_graph of each substation
        factored_observation: Dict[
            Substation, Optional[dgl.DGLHeteroGraph]
        ] = factor_observation(
            observation_graph,
            str(self.device),
            self.architecture.pop.agent_neighbourhood_radius,
        )

        return factored_observation, observation_graph

    def step(
        self,
        action: EncodedAction,  # kept for interface consistency
        observation: BaseObservation,
        reward: float,
        next_observation: BaseObservation,
        done: bool,
    ):
        if done:
            self.log_alive_steps(self.alive_steps, self.episodes)
            self.train_steps += 1
            self.episodes += 1
            self.alive_steps = 0
            self.old_graph = (
                None  # this resets the incremental community detection algorithm
            )
        else:
            # Log reward to tensorboard
            repeated_action_penalty: float = self.action_detector.penalty()
            reward += repeated_action_penalty
            self.log_reward(reward, self.train_steps)
            self.log_penalty(repeated_action_penalty, self.train_steps)
            self.alive_steps += 1
            self.train_steps += 1

            # Normal graph of the next_observation
            next_graph: nx.Graph = next_observation.as_networkx()

            # Community structure is updated and managers are assigned to the new communities
            # By taking the most similar communities wrt the old mappings
            next_communities, next_community_to_manager = self._update_communities(
                observation.as_networkx(), next_graph
            )

            # Factor next_observation for each agent
            next_factored_observation: Dict[
                Substation, Optional[dgl.DGLHeteroGraph]
            ] = factor_observation(
                next_graph,
                device=str(self.device),
                radius=self.architecture.pop.agent_neighbourhood_radius,
            )

            # Stop the decay of the agent whose action has been selected
            # In case no-action is selected multiple agents may have their decay stopped
            agents_stop_decay: Dict[Substation, bool] = (
                {
                    sub_id: False if agent_action == self.chosen_action else True
                    for sub_id, agent_action in self.substation_to_encoded_action.items()
                }
                if self.architecture.pop.epsilon_beta_scheduling
                else {
                    sub_id: False for sub_id in self.substation_to_encoded_action.keys()
                }
            )

            # Query agents for action
            next_substation_to_local_actions: Dict[
                Substation, int
            ] = self._get_agent_actions(next_factored_observation)

            # Factor next_observation given the new community structure
            (
                next_sub_graphs,
                next_substation_to_encoded_action,
            ) = self._compute_managers_sub_graphs(
                graph=next_graph,
                substation_to_local_action=next_substation_to_local_actions,
                new_communities=next_communities,
            )

            # Stop the decay of the managers whose action has been selected
            # In case no-action is selected multiple agents may have their decay stopped
            manager_stop_decay: Dict[Community, bool] = (
                {
                    community: False if community == self.chosen_community else True
                    for community in self.community_to_manager.keys()
                }
                if self.architecture.pop.epsilon_beta_scheduling
                else {
                    community: False for community in self.community_to_manager.keys()
                }
            )

            # Children may add needed functionalities to the step function by extending extra_step
            self._extra_step(
                next_sub_graphs=next_sub_graphs,
                next_graph=next_graph,
                next_substation_to_encoded_action=next_substation_to_encoded_action,
                reward=reward,
                action=self.chosen_node,
                done=done,
                next_communities=next_communities,
                next_community_to_manager=next_community_to_manager,
            )

            self._step_managers(
                community_to_sub_graphs_dict=self.sub_graphs,
                actions=self.community_to_substation,
                reward=reward,
                next_community_to_sub_graphs_dict=next_sub_graphs,
                done=done,
                stop_decay=manager_stop_decay,
                next_community_to_manager=next_community_to_manager,
            )

            self._step_agents(
                factored_observation=self.factored_observation,
                actions=self.substation_to_local_action,
                reward=reward,
                next_factored_observation=next_factored_observation,
                done=done,
                stop_decay=agents_stop_decay,
            )

    def get_state(self: "BasePOP") -> Dict[str, Any]:
        """
        Get System state
        Children may use this as a starting point for reporting system state
        """

        agents_state: List[Dict[str, Any]] = ray.get(
            [agent.get_state.remote() for _, agent in self.substation_to_agent.items()]
        )

        managers_state: List[Dict[str, Any]] = ray.get(
            [manager.get_state.remote() for manager in self.managers_history.keys()]
        )
        managers_name: List[str] = ray.get(
            [manager.get_name.remote() for manager in self.managers_history.keys()]
        )
        return {
            "agents_state": {
                sub_id: agent_state
                for sub_id, agent_state in zip(
                    list(self.substation_to_agent.keys()), agents_state
                )
            },
            "managers_state": {
                manager_name: (state, self.managers_history[manager])
                for manager, manager_name, state in zip(
                    list(self.managers_history.keys()), managers_name, managers_state
                )
            },
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "train_steps": self.train_steps,
            "episodes": self.episodes,
            "alive_steps": self.alive_steps,
            "community_update_steps": self.community_update_steps,
            "manager_initialization_threshold": self.manager_initialization_threshold,
            "name": self.name,
            "architecture": asdict(self.architecture),
            "seed": self.seed,
            "device": str(self.device),
        }

    def _get_substation_to_agent_mapping(
        self, substation_to_action_space: Dict[Substation, List[int]]
    ) -> Dict[int, IdToAct]:
        """
        Map one converter to each substation
        Converters are used to map encoded actions to global actions

        """

        mapping: Dict[int, IdToAct] = {}
        for sub_id, action_space in substation_to_action_space.items():
            conv = IdToAct(self.env.action_space)
            conv.init_converter(action_space)
            conv.seed(self.seed)
            mapping[sub_id] = conv
        return mapping

    def retrieve_promises_batched(self, promises: list, batch_size: int) -> list:
        answers: list = []
        for i in range(0, len(promises), batch_size):
            answers.extend(ray.get(promises[i : i + batch_size]))
        return answers

    def _get_agent_actions(
        self, factored_observation: Dict[Substation, Optional[dgl.DGLHeteroGraph]]
    ) -> Dict[Substation, int]:
        """
        Query each agent for 1 action
        """
        no_action_positions_to_add: List[int] = []

        # Observations are None in case the neighbourhood is empty (e.g. isolated nodes)
        # In such case no_action (id = 0) is selected
        # Each agent returns an action for its associated Substation

        actions = ray.get(
            list(
                filter(
                    lambda x: x is not None,
                    [
                        self.substation_to_agent[sub_id].take_action.remote(
                            observation,
                        )
                        if observation is not None
                        else no_action_positions_to_add.append(idx)
                        for idx, (sub_id, observation) in enumerate(
                            factored_observation.items()
                        )
                    ],
                )
            )
        )

        # no_action is added for each None neighbourhood
        for no_action_position in no_action_positions_to_add:
            actions.insert(no_action_position, 0)

        return {
            sub_id: action
            for sub_id, action in zip(factored_observation.keys(), actions)
        }

    def _get_manager_actions(
        self,
        community_to_sub_graphs_dict: Dict[Community, dgl.DGLHeteroGraph],
        substation_to_encoded_action: Dict[Substation, EncodedAction],
        communities: List[Community],
        community_to_manager: Dict[Community, Manager],
    ) -> Dict[Community, Substation]:
        """
        Query one action per community
        """

        no_action_positions_to_add: List[int] = []

        # Managers are queried for an action
        # Each manager chooses one action for each community she handles
        # In case of single node communities managers choose no action
        actions: List[Substation] = ray.get(
            list(
                filter(
                    lambda x: x is not None,
                    [
                        community_to_manager[community].take_action.remote(
                            transformed_observation=community_to_sub_graphs_dict[
                                community
                            ],
                            mask=frozenset(
                                [
                                    sub
                                    for sub in community
                                    if sub in list(substation_to_encoded_action.keys())
                                ]
                            ),
                        )
                        if community_to_sub_graphs_dict[community].num_edges() > 0
                        else no_action_positions_to_add.append(idx)
                        for idx, community in enumerate(communities)
                    ],
                )
            )
        )

        # no_action is added for each single node community
        for no_action_position in no_action_positions_to_add:
            actions.insert(no_action_position, 0)

        return {community: action for community, action in zip(communities, actions)}

    def _compute_managers_sub_graphs(
        self,
        graph: nx.graph,
        substation_to_local_action: Dict[Substation, int],
        new_communities: Optional[List[Community]] = None,
    ) -> Tuple[Dict[Community, dgl.DGLHeteroGraph], Dict[Substation, EncodedAction]]:
        substation_to_encoded_action: Dict[Substation, EncodedAction] = {
            sub_id: self._lookup_local_action(
                self.substation_to_action_converter[sub_id].all_actions[local_action]
            )
            for sub_id, local_action in substation_to_local_action.items()
        }

        # Each agent is assigned to its chosen action
        nx.set_node_attributes(
            graph,
            {
                node_id: {
                    "action": substation_to_encoded_action[node_data["sub_id"]],
                }
                for node_id, node_data in graph.nodes.data()
            },
        )

        return (
            split_graph_into_communities(
                graph,
                self.communities if new_communities is None else new_communities,
                str(self.device),
            ),
            substation_to_encoded_action,
        )

    def _compute_summarized_graph(
        self,
        graph,
        sub_graphs: Dict[Community, dgl.DGLHeteroGraph],
        substation_to_encoded_action: Dict[Substation, EncodedAction],
        community_to_substation: Dict[Community, Substation],
        new_communities: Optional[List[Community]] = None,
        new_community_to_manager_dict: Optional[Dict[Community, Manager]] = None,
    ):
        # The graph is summarized by contracting every community in 1 supernode
        # And storing the embedding of each manager in each supernode as node feature
        # Together with the action chosen by the manager
        return self._summarize_graph(
            graph,
            {
                community: substation_to_encoded_action[substation]
                for community, substation in community_to_substation.items()
            },
            sub_graphs,
            new_communities=new_communities,
            new_community_to_manager_dict=new_community_to_manager_dict,
        ).to(self.device)

    @staticmethod
    def _jaccard_distance(s1: frozenset, s2: frozenset) -> float:
        _s1 = set(s1)
        _s2 = set(s2)
        return len(_s1.intersection(_s2)) / len(_s1.union(_s2))

    @staticmethod
    def exponential_decay(initial_value: float, half_life: float, t: float):
        return initial_value * 2 ** (-t / half_life) if half_life != 0 else 0

    def _initialize_new_manager(
        self, new_communities: List[Community]
    ) -> Dict[Community, Manager]:
        if self.manager_initialization_threshold <= 1 / (self.env.n_sub * 2):
            # This is the minimum min_max_jaccard, so we skip computation completely when we reach it
            return {}
        self.community_update_steps += 1
        self.manager_initialization_threshold = self.exponential_decay(
            initial_value=1,  # Start from the maximum jaccard value
            half_life=self.architecture.pop.manager_initialization_half_life,
            t=self.community_update_steps,
        )
        if not self.managers_history:
            # If no manager exists create one for each detected community
            managers = [
                Manager.remote(
                    agent_actions=self.env.n_sub * 2,
                    node_features=self.node_features + 1,  # Node Features + Action
                    edge_features=self.edge_features,
                    architecture=self.architecture.manager,
                    name="manager_" + str(idx) + "_" + self.name,
                    training=self.training,
                    device=self.device,
                )
                for idx in range(len(new_communities))
            ]

            self.managers_history = {
                manager: FixedSet(int(self.architecture.pop.manager_history_size))
                for manager in managers
            }
            return {
                community: manager
                for community, manager in zip(new_communities, managers)
            }
        else:
            most_distant_community, min_max_jaccard = min(
                [
                    (
                        community,
                        max(
                            [
                                max(
                                    [
                                        self._jaccard_distance(old_community, community)
                                        for old_community in community_history
                                    ]
                                )
                                for manager, community_history in self.managers_history.items()
                            ]
                        ),
                    )
                    for community in new_communities
                ],
                key=lambda x: x[1],
            )
            if min_max_jaccard < self.manager_initialization_threshold:
                manager = Manager.remote(
                    agent_actions=self.env.n_sub * 2,
                    node_features=self.node_features + 1,  # Node Features + Action
                    edge_features=self.edge_features,
                    architecture=self.architecture.manager,
                    name="manager_" + str(len(self.managers_history)) + "_" + self.name,
                    training=self.training,
                    device=self.device,
                )
                self.managers_history[manager] = FixedSet(
                    int(self.architecture.pop.manager_history_size)
                )
                return {most_distant_community: manager}
            else:
                return {}

    def _update_communities(
        self, graph: nx.Graph, next_graph: Optional[nx.Graph] = None
    ) -> Tuple[List[Community], Dict[Community, Manager]]:
        if next_graph is None:
            new_communities = self.community_detector.dynamo(graph_t=graph)
        else:
            new_communities = self.community_detector.dynamo(
                graph_t=graph,
                graph_t1=next_graph,
                comm_t=self.communities,
            )
        self.log_communities(new_communities, self.train_steps)

        if self.communities and set(self.communities) == set(new_communities):
            updated_community_to_manager_dict: Dict[
                Community, Manager
            ] = self._initialize_new_manager(new_communities)
            return new_communities, {
                **self.community_to_manager,
                **updated_community_to_manager_dict,
            }

        updated_community_to_manager_dict: Dict[
            Community, Manager
        ] = self._initialize_new_manager(new_communities)

        for community, manager in updated_community_to_manager_dict.items():
            self.managers_history[manager].add(community)

        for new_community in filter(
            lambda x: x not in updated_community_to_manager_dict.keys(), new_communities
        ):
            manager_to_jaccard_dict = {
                manager: max(
                    [
                        self._jaccard_distance(old_community, new_community)
                        for old_community in communities
                    ]
                )
                for manager, communities in self.managers_history.items()
            }
            updated_community_to_manager_dict[new_community] = max(
                manager_to_jaccard_dict, key=manager_to_jaccard_dict.get
            )
            self.managers_history[updated_community_to_manager_dict[new_community]].add(
                new_community
            )

        return new_communities, updated_community_to_manager_dict

    def _step_managers(
        self,
        community_to_sub_graphs_dict: Dict[Community, dgl.DGLHeteroGraph],
        actions: Dict[Community, Substation],
        reward: float,
        next_community_to_sub_graphs_dict: Dict[Community, dgl.DGLHeteroGraph],
        done: bool,
        stop_decay: Dict[Community, bool],
        next_community_to_manager: Dict[Community, Manager],
    ) -> None:

        # Build a mapping between each manager and the communities she handles
        # The mapping represents communities before and after applying the chosen action
        manager_to_community_transformation: Dict[
            Manager, Tuple[List[Community], List[Community]]
        ] = {manager: ([], []) for manager in self.managers_history.keys()}
        for community, manager in self.community_to_manager.items():
            manager_to_community_transformation[manager][0].append(community)

        for new_community, manager in next_community_to_manager.items():
            manager_to_community_transformation[manager][1].append(new_community)

        # Step the managers
        # For each community before applying the action
        # Find the most similar community among the ones managed after applying the action
        losses, rewards = zip(
            *ray.get(
                list(
                    itertools.chain(
                        *[
                            [
                                manager.step.remote(
                                    observation=community_to_sub_graphs_dict[
                                        old_community
                                    ],
                                    action=actions[old_community],
                                    reward=reward,
                                    next_observation=next_community_to_sub_graphs_dict[
                                        new_manager_communities[
                                            np.argmax(
                                                [
                                                    self._jaccard_distance(
                                                        old_community,
                                                        new_community,
                                                    )
                                                    for new_community in new_manager_communities
                                                    if next_community_to_sub_graphs_dict[
                                                        new_community
                                                    ].num_edges()
                                                    > 0
                                                ]
                                            )
                                        ]
                                    ],
                                    done=done,
                                    stop_decay=stop_decay[old_community],
                                )
                                for old_community in old_manager_communities
                                if community_to_sub_graphs_dict[
                                    old_community
                                ].num_edges()
                                > 0
                            ]
                            for manager, (
                                old_manager_communities,
                                new_manager_communities,
                            ) in manager_to_community_transformation.items()
                            if list(
                                filter(
                                    lambda x: community_to_sub_graphs_dict[
                                        x
                                    ].num_edges()
                                    > 0,
                                    old_manager_communities,
                                )
                            )
                            and list(
                                filter(
                                    lambda x: next_community_to_sub_graphs_dict[
                                        x
                                    ].num_edges()
                                    > 0,
                                    new_manager_communities,
                                )
                            )
                        ]
                    )
                )
            )
        )
        names = ray.get(
            [
                self.community_to_manager[community].get_name.remote()
                for community in self.communities
            ]
        )

        self.log_step(
            losses=losses,
            implicit_rewards=[full_reward - reward for full_reward in rewards],
            names=names,
            train_steps=self.train_steps,
        )

    def _step_agents(
        self,
        factored_observation: Dict[Substation, dgl.DGLHeteroGraph],
        actions: Dict[Substation, int],
        reward: float,
        next_factored_observation: Dict[Substation, dgl.DGLHeteroGraph],
        done: bool,
        stop_decay: Dict[Substation, bool],
    ) -> None:
        # List of promises to get with ray.get
        step_promises = []

        # Substations present in next_factored_observation
        substations = []
        for sub_id, observation in factored_observation.items():
            next_observation: Optional[
                dgl.DGLHeteroGraph
            ] = next_factored_observation.get(sub_id)
            if next_observation is not None:

                # Build step promise with the previous and next observation
                step_promises.append(
                    self.substation_to_agent[sub_id].step.remote(
                        observation=observation,
                        action=actions[sub_id],
                        reward=reward,
                        next_observation=next_factored_observation[sub_id],
                        done=done,
                        stop_decay=stop_decay[sub_id],
                    )
                )
                substations.append(sub_id)
            else:
                print(
                    "("
                    + str(self.train_steps)
                    + ") Substation '"
                    + str(sub_id)
                    + "' not present in current next_state, related agent is not stepping..."
                )

        # Step the agents
        losses, rewards = zip(*ray.get(step_promises))
        names = ray.get(
            [
                self.substation_to_agent[substation].get_name.remote()
                for substation in substations
            ]
        )
        self.log_step(
            losses=losses,
            implicit_rewards=[full_reward - reward for full_reward in rewards],
            names=names,
            train_steps=self.train_steps,
        )

    def _lookup_local_action(self, action: BaseAction):
        return self.action_lookup_table[HashableAction(action)]

    def _summarize_graph(
        self,
        graph: nx.Graph,
        manager_actions: Dict[Community, EncodedAction],
        sub_graphs: Dict[Community, dgl.DGLHeteroGraph],
        new_communities: Optional[List[Community]] = None,
        new_community_to_manager_dict: Optional[Dict[Community, Manager]] = None,
    ) -> dgl.DGLHeteroGraph:

        current_community_to_manager_dict = (
            self.community_to_manager
            if new_community_to_manager_dict is None
            else new_community_to_manager_dict
        )

        node_attribute_dict = {}
        for community, sub_graph in sub_graphs.items():
            node_embeddings = (
                ray.get(
                    current_community_to_manager_dict[
                        community
                    ].get_node_embeddings.remote(sub_graph)
                )
                if sub_graph.num_nodes() > 1
                else th.zeros(
                    (
                        1,
                        ray.get(
                            current_community_to_manager_dict[
                                community
                            ].get_embedding_size.remote()
                        ),
                    )
                )
            )
            sub_graphs[community].ndata["node_embeddings"] = node_embeddings
            for node in community:
                node_attribute_dict[node] = {
                    "embedding": dgl.mean_nodes(
                        sub_graphs[community], "node_embeddings"
                    )
                    .squeeze()
                    .detach(),
                }
            del sub_graphs[community].ndata["node_embeddings"]

        nx.set_node_attributes(
            graph,
            node_attribute_dict,
        )

        # Graph is summarized by contracting communities into supernodes
        # Manager action is assigned to each supernode
        summarized_graph: nx.graph = graph
        for community in (
            self.communities if new_communities is None else new_communities
        ):
            community_list = list(community)
            for node in community_list[1:]:
                summarized_graph = nx.contracted_nodes(
                    summarized_graph, community_list[0], node
                )
            summarized_graph.nodes[community_list[0]]["embedding_action"] = th.cat(
                (
                    summarized_graph.nodes[community_list[0]]["embedding"],
                    th.tensor(
                        [
                            1 if substation in community else 0
                            for substation in range(self.env.n_sub)
                        ]
                    ),
                    th.tensor([manager_actions[community]]),
                ),
                dim=-1,
            )

        # The summarized graph is returned in DGL format
        # Each supernode has the action chosen by its community manager
        # And the contracted embedding
        return dgl.from_networkx(
            summarized_graph.to_directed(),
            node_attrs=["embedding_action"],
            device=self.device,
        )


def train(env: BaseEnv, iterations: int, dpop, save_frequency: int = 3600):

    training_step: int = 0
    obs: BaseObservation = env.reset()
    done = False
    reward = env.reward_range[0]
    total_episodes = len(env.chronics_handler.subpaths)

    last_save_time = time.time()
    print("Model will be checkpointed every " + str(save_frequency) + " seconds")
    with tqdm(total=iterations - training_step) as pbar:
        while training_step < iterations:
            if dpop.episodes % total_episodes == 0:
                env.chronics_handler.shuffle()
            if done:
                obs = env.reset()
            encoded_action = dpop.my_act(dpop.convert_obs(obs), reward, done)
            action = dpop.convert_act(encoded_action)
            next_obs, reward, done, _ = env.step(action)
            dpop.step(
                action=encoded_action,
                observation=obs,
                reward=reward,
                next_observation=next_obs,
                done=done,
            )
            obs = next_obs
            training_step += 1

            current_time = time.time()
            if current_time - last_save_time >= save_frequency:
                # Every Hour Save the model
                print("Saving Checkpoint")
                dpop.save()
                last_save_time = time.time()

            pbar.update(1)

    print("\nSaving\n")

    dpop.save()
    ray.shutdown()
