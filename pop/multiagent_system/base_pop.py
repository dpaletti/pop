import itertools
from pathlib import Path
import time
from abc import abstractmethod
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

import dgl
import networkx as nx
import numpy as np
import ray
from ray.util.client.common import ClientActorHandle
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
    generate_redispatching_action_space,
    factor_observation,
    split_graph_into_communities,
)
from pop.networks.serializable_module import SerializableModule
from pop.multiagent_system.dictatorship_penalizer import DictatorshipPenalizer

from pop.multiagent_system.reward_distributor import Incentivizer
import random


class BasePOP(AgentWithConverter, SerializableModule, LoggableModule):
    def __init__(
        self,
        env: BaseEnv,
        name: str,
        architecture: Architecture,
        training: bool,
        seed: int,
        feature_ranges: Dict[str, Tuple[float, float]],
        checkpoint_dir: Optional[str] = None,
        tensorboard_dir: Optional[str] = None,
        device: Optional[str] = None,
        pre_train: bool = False,
    ):
        AgentWithConverter.__init__(self, env.action_space, IdToAct)
        SerializableModule.__init__(self, checkpoint_dir, name)
        LoggableModule.__init__(self, tensorboard_dir)

        self.name = name
        self.seed: int = seed
        self.env = env
        self.node_features: List[str] = architecture.pop.node_features
        self.edge_features: List[str] = architecture.pop.edge_features
        self.feature_ranges = feature_ranges
        self.pre_train = pre_train

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

        # Action Space Initialization
        if self.architecture.pop.generator_storage_only:
            (
                substation_to_action_space,
                self.action_lookup_table,
            ) = generate_redispatching_action_space(
                self.env, self.architecture.pop.actions_per_generator
            )
        else:

            substation_to_action_space, self.action_lookup_table = factor_action_space(
                env.observation_space,
                self.converter,
                self.env.n_sub,
                composite_actions=self.architecture.pop.composite_actions,
                generator_storage_only=self.architecture.pop.generator_storage_only,
                remove_no_action=self.architecture.pop.remove_no_action,
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
                feature_ranges=feature_ranges,
            )
            if len(action_space) > 1
            else RayShallowGCNAgent(
                name="agent_" + str(sub_id) + "_" + self.name,
                device=self.device,
            )
            for sub_id, action_space in substation_to_action_space.items()
        }
        # Managers
        self.manager_feature_ranges = {
            "node_features": {
                **feature_ranges["node_features"],
                "action": tuple(
                    [
                        0,
                        max(
                            [
                                len(action_space)
                                for action_space in substation_to_action_space.values()
                            ]
                        ),
                    ]
                ),
            },
            "edge_features": feature_ranges["edge_features"],
        }
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

        if self.architecture.pop.incentives:
            self.agent_incentives: Incentivizer = Incentivizer(
                {
                    substation: len(action_converter.all_actions)
                    for substation, action_converter in self.substation_to_action_converter.items()
                },
                **self.architecture.pop.incentives
            )
            self.manager_incentives: Incentivizer = Incentivizer(
                {}, **self.architecture.pop.incentives
            )

        if self.architecture.pop.dictatorship_penalty:
            self.manager_dictatorship_penalties: Dict[
                Manager, DictatorshipPenalizer
            ] = {}

    @abstractmethod
    def get_action(
        self, observation: dgl.DGLHeteroGraph
    ) -> Tuple[int, Optional[float]]:
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
        (
            self.substation_to_local_action,
            substation_to_q_values,
        ) = self._get_agent_actions(self.factored_observation)

        # Split graph into communities
        (
            self.sub_graphs,
            self.substation_to_encoded_action,
        ) = self._compute_managers_sub_graphs(
            graph=graph, substation_to_local_action=self.substation_to_local_action
        )

        # Managers chooses the best substation for each community they handle
        self.community_to_substation, community_to_q_values = self._get_manager_actions(
            graph,
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
        self.chosen_node, chosen_node_q_value = self.get_action(self.summarized_graph)
        chosen_node_features = self.summarized_graph.ndata[
            self.architecture.pop.head_manager_embedding_name
        ][self.chosen_node]
        self.chosen_action = int(chosen_node_features[-1].item())
        self.chosen_community = frozenset(
            [
                idx
                for idx, one_hot in enumerate(
                    chosen_node_features[-(1 + self.env.n_sub * 2) : -1].tolist()
                )
                if one_hot == 1
            ]
        )

        manager_names: List[str] = ray.get(
            [
                manager.get_name.remote()
                for manager in list(self.community_to_manager.values())
            ]
        )
        manager_names: List[str] = [
            "_".join(name.split("_")[0:2]) for name in manager_names
        ]
        community_to_names: Dict[Community, str] = {
            community: manager_names[idx]
            for idx, community in enumerate(list(self.community_to_manager.keys()))
        }

        agent_names: Dict[Substation, str] = {
            sub: "_".join(ray.get(agent.get_name.remote()).split("_")[0:2])
            for sub, agent in self.substation_to_agent.items()
            if type(agent) is ClientActorHandle
        }

        if self.action_detector.is_repeated(self.chosen_action):
            self.chosen_action = 0

        # Log to Tensorboard
        self.log_system_behaviour(
            best_action=self.chosen_action,
            head_manager_action=self.chosen_node,
            head_manager_q_value=chosen_node_q_value,
            manager_actions={
                community: (action, community_to_names[community])
                for community, action in self.community_to_substation.items()
            },
            manager_q_values={
                community: (action, community_to_names[community])
                for community, action in community_to_q_values.items()
            }
            if community_to_q_values
            else None,
            agent_actions={
                agent_names[sub]: action
                for sub, action in self.substation_to_local_action.items()
            },
            agent_q_values={
                agent_names[sub]: q_value
                for sub, q_value in substation_to_q_values.items()
            }
            if substation_to_q_values
            else None,
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
                    agent_names.values(),
                    ray.get(
                        [
                            agent.get_exploration_logs.remote()
                            for agent in self.substation_to_agent.values()
                            if type(agent) is ClientActorHandle
                        ]
                    ),
                )
            },
            train_steps=self.train_steps,
        )

        if not self.training:
            self.train_steps += 1

        return self.chosen_action

    def convert_obs(
        self, observation: BaseObservation
    ) -> Tuple[Dict[Substation, dgl.DGLHeteroGraph], nx.Graph]:
        """
        Upon calling act(observation)
        The agent calls my_act(convert_obs(observation))
        This method is thus an implicit observation converter for the agent.
        Inherited from AgentWithConverter (Grid2Op)
        """

        # Observation as a "normal graph"
        observation_graph: nx.Graph = observation.as_networkx()

        # Observation is factored for each agent by taking the ego_graph of each substation
        factored_observation: Dict[Substation, dgl.DGLHeteroGraph] = factor_observation(
            obs_graph=observation_graph,
            node_features=self.node_features,
            edge_features=self.edge_features,
            device=str(self.device),
            radius=self.architecture.pop.agent_neighbourhood_radius,
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
        # Log reward to tensorboard
        repeated_action_penalty: float = self.action_detector.penalty()
        self.log_reward(reward, self.train_steps, name="Reward")
        penalized_reward: float = reward + repeated_action_penalty
        self.log_reward(penalized_reward, self.train_steps, name="Penalized Reward")
        self.log_reward(
            repeated_action_penalty,
            self.train_steps,
            name="Repeated Action Penalty",
        )
        self.alive_steps += 1
        self.train_steps += 1

        selected_substations: List[Substation] = [
            sub
            for sub, agent_action in self.substation_to_encoded_action.items()
            if agent_action == self.chosen_action
        ]

        # Stop the decay of the agent whose action has been selected
        # In case no-action is selected multiple agents may have their decay stopped
        agents_stop_decay: Dict[Substation, bool] = (
            {
                sub_id: False if sub_id in selected_substations else True
                for sub_id, agent_action in self.substation_to_encoded_action.items()
            }
            if self.architecture.pop.epsilon_beta_scheduling
            else {sub_id: False for sub_id in self.substation_to_encoded_action.keys()}
        )

        # Stop the decay of the managers whose action has been selected
        # In case no-action is selected multiple agents may have their decay stopped
        manager_stop_decay: Dict[Community, bool] = (
            {
                community: False if community == self.chosen_community else True
                for community in self.community_to_manager.keys()
            }
            if self.architecture.pop.epsilon_beta_scheduling
            else {community: False for community in self.community_to_manager.keys()}
        )

        current_agent_incentives: Optional[Dict[Substation, float]] = None
        current_manager_incentives: Optional[Dict[Manager, float]] = None
        if self.architecture.pop.incentives:
            current_agent_incentives = self.agent_incentives.incentives(
                selected_substations  # type: ignore
            )
            current_manager_incentives = self.manager_incentives.incentives(
                [self.community_to_manager[self.chosen_community]]  # type: ignore
            )  # type: ignore

        current_manager_dictatorship_penalty: Optional[Dict[Manager, float]] = None

        if self.architecture.pop.dictatorship_penalty:
            current_manager_dictatorship_penalty = {
                manager: 0 for manager in self.manager_dictatorship_penalties
            }
            for (
                community,
                chosen_substation,
            ) in self.community_to_substation.items():
                manager = self.community_to_manager[community]
                current_manager_dictatorship_penalty[
                    manager
                ] += self.manager_dictatorship_penalties[manager].penalty(
                    chosen_substation
                )

        if done:
            self.log_alive_steps(self.alive_steps, self.episodes)
            self.train_steps += 1
            self.episodes += 1
            self.alive_steps = 0
            self.old_graph = (
                None  # this resets the incremental community detection algorithm
            )
            if self.architecture.pop.incentives:
                self.agent_incentives.reset()
                self.manager_incentives.reset()

            if self.architecture.pop.dictatorship_penalty:
                for manager, penalty in self.manager_dictatorship_penalties.items():
                    penalty.reset()
                try:
                    # in case of DPOP
                    self.dictatorship_penalty.reset()
                except:
                    pass

            self._extra_step(
                next_sub_graphs=None,
                next_graph=None,
                next_substation_to_encoded_action=None,
                reward=penalized_reward,
                action=self.chosen_node,
                done=done,
                next_communities=None,
                next_community_to_manager=None,
            )

            self._step_managers(
                community_to_sub_graphs_dict=self.sub_graphs,
                actions=self.community_to_substation,
                reward=reward,
                next_community_to_sub_graphs_dict=None,
                done=done,
                stop_decay=manager_stop_decay,
                next_community_to_manager=None,
                chosen_communities=[self.chosen_community]
                if self.architecture.pop.manager_selective_learning
                else self.communities,
                incentives=current_manager_incentives,
                dictatorship_penalties=current_manager_dictatorship_penalty,
            )

            self._step_agents(
                factored_observation=self.factored_observation,
                actions=self.substation_to_local_action
                if not self.architecture.pop.no_action_reward
                else {
                    substation: local_action
                    if substation in selected_substations
                    else 0
                    for substation, local_action in self.substation_to_local_action.items()
                },
                reward=reward,
                next_factored_observation=None,
                done=done,
                stop_decay=agents_stop_decay,
                selected_substations=selected_substations
                if self.architecture.pop.agent_selective_learning
                else list(self.substation_to_local_action.keys()),
                incentives=current_agent_incentives,
            )

        else:

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
                obs_graph=next_graph,
                node_features=self.node_features,
                edge_features=self.edge_features,
                device=str(self.device),
                radius=self.architecture.pop.agent_neighbourhood_radius,
            )

            # Query agents for action
            next_substation_to_local_actions, _ = self._get_agent_actions(
                next_factored_observation
            )

            # Factor next_observation given the new community structure
            (
                next_sub_graphs,
                next_substation_to_encoded_action,
            ) = self._compute_managers_sub_graphs(
                graph=next_graph,
                substation_to_local_action=next_substation_to_local_actions,
                new_communities=next_communities,
            )

            # Children may add needed functionalities to the step function by extending extra_step
            self._extra_step(
                next_sub_graphs=next_sub_graphs,
                next_graph=next_graph,
                next_substation_to_encoded_action=next_substation_to_encoded_action,
                reward=penalized_reward,
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
                chosen_communities=[self.chosen_community]
                if self.architecture.pop.manager_selective_learning
                else self.communities,
                incentives=current_manager_incentives,
                dictatorship_penalties=current_manager_dictatorship_penalty,
            )

            self._step_agents(
                factored_observation=self.factored_observation,
                actions=self.substation_to_local_action
                if not self.architecture.pop.no_action_reward
                else {
                    substation: local_action
                    if substation in selected_substations
                    else 0
                    for substation, local_action in self.substation_to_local_action.items()
                },
                reward=reward,
                next_factored_observation=next_factored_observation,
                done=done,
                stop_decay=agents_stop_decay,
                selected_substations=selected_substations
                if self.architecture.pop.agent_selective_learning
                else list(self.substation_to_local_action.keys()),
                incentives=current_agent_incentives,
            )

    def get_state(self: "BasePOP") -> Dict[str, Any]:
        """
        Get System state
        Children may use this as a starting point for reporting system state
        """

        agents_state: List[Dict[str, Any]] = [
            ray.get(agent.get_state.remote())
            if type(agent) is ClientActorHandle
            else agent.get_state()
            for _, agent in self.substation_to_agent.items()
        ]

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
            "feature_ranges": self.feature_ranges,
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
        self, factored_observation: Dict[Substation, dgl.DGLHeteroGraph]
    ) -> Tuple[Dict[Substation, int], Optional[Dict[Substation, float]]]:
        """
        Query each agent for 1 action
        """
        # Observations are None in case the neighbourhood is empty (e.g. isolated nodes)
        # In such case no_action (id = 0) is selected
        # Each agent returns an action for its associated Substation

        if self.pre_train:
            return {substation: 0 for substation in factored_observation.keys()}, None

        actions_taken = ray.get(
            [
                self.substation_to_agent[sub_id].take_action.remote(observation)
                for sub_id, observation in factored_observation.items()
                if type(self.substation_to_agent[sub_id]) is ClientActorHandle
            ]
        )

        actions, q_values = zip(*actions_taken)

        return {
            sub_id: action
            for sub_id, action in zip(
                [
                    sub
                    for sub in factored_observation.keys()
                    if type(self.substation_to_agent[sub]) is ClientActorHandle
                ],
                actions,
            )
        }, {
            sub_id: q_value
            for sub_id, q_value in zip(
                [
                    sub
                    for sub in factored_observation.keys()
                    if type(self.substation_to_agent[sub]) is ClientActorHandle
                ],
                q_values,
            )
        }

    def _get_manager_actions(
        self,
        graph: nx.Graph,
        community_to_sub_graphs_dict: Dict[Community, dgl.DGLHeteroGraph],
        substation_to_encoded_action: Dict[Substation, EncodedAction],
        communities: List[Community],
        community_to_manager: Dict[Community, Manager],
    ) -> Tuple[Dict[Community, Substation], Optional[Dict[Community, float]]]:
        """
        Query one action per community
        """
        if self.pre_train:
            return {
                community: random.sample(community, 1)[0]
                for community in community_to_sub_graphs_dict.keys()
            }, None

        # Managers are queried for an action
        # Each manager chooses one action for each community she handles

        enabled_nodes = [
            node
            for community in communities
            for node in community
            if not self.architecture.pop.manager_remove_no_action
            or len(
                self.substation_to_action_converter[
                    graph.nodes[node]["sub_id"]
                ].all_actions
            )
            > 1
        ]

        actions, q_values = zip(
            *ray.get(
                [
                    community_to_manager[community].take_action.remote(
                        transformed_observation=community_to_sub_graphs_dict[community],
                        mask=frozenset(
                            [node for node in community if node in enabled_nodes]
                            if set(community).intersection(set(enabled_nodes))
                            else [list(community)[0]]
                        ),
                    )
                    for community in communities
                ],
            )
        )

        return {
            community: graph.nodes.data()[action]["sub_id"]
            for community, action in zip(communities, actions)
        }, {community: q_value for community, q_value in zip(communities, q_values)}

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
                    "action": substation_to_encoded_action[node_data["sub_id"]]
                    if substation_to_encoded_action.get(node_data["sub_id"])
                    else 0,
                }
                for node_id, node_data in graph.nodes.data()
            },
        )

        return (
            split_graph_into_communities(
                graph=graph,
                node_features=self.node_features + ["action"],
                edge_features=self.edge_features,
                communities=self.communities
                if new_communities is None
                else new_communities,
                device=str(self.device),
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
                if substation_to_encoded_action.get(substation)
                else 0
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
                    node_features=self.node_features + ["action"],
                    edge_features=self.edge_features,
                    architecture=self.architecture.manager,
                    name="manager_" + str(idx) + "_" + self.name,
                    training=self.training,
                    device=self.device,
                    feature_ranges=self.manager_feature_ranges,
                )
                for idx in range(len(new_communities))
            ]

            if self.architecture.pop.incentives:
                # TODO: find a way to model manager importance, for now they are all the same
                # TODO: care that manager importance may change with time due to dynamic community structure
                for manager in managers:
                    self.manager_incentives.add_agent(manager, 1)

            if self.architecture.pop.dictatorship_penalty:
                for manager in managers:
                    self.manager_dictatorship_penalties[
                        manager
                    ] = DictatorshipPenalizer(
                        choice_to_ranking={
                            substation: len(action_converter.all_actions)
                            for substation, action_converter in self.substation_to_action_converter.items()
                        },
                        **self.architecture.pop.dictatorship_penalty
                    )

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
                    node_features=self.node_features + ["action"],
                    edge_features=self.edge_features,
                    architecture=self.architecture.manager,
                    name="manager_" + str(len(self.managers_history)) + "_" + self.name,
                    training=self.training,
                    device=self.device,
                    feature_ranges=self.manager_feature_ranges,
                )
                self.managers_history[manager] = FixedSet(
                    int(self.architecture.pop.manager_history_size)
                )
                if self.architecture.pop.incentives:
                    # TODO: find a way to model manager importance, for now they are all the same
                    self.manager_incentives.add_agent(manager, 1)

                if self.architecture.pop.dictatorship_penalty:
                    self.manager_dictatorship_penalties[
                        manager
                    ] = DictatorshipPenalizer(
                        choice_to_ranking={
                            substation: len(action_converter.all_actions)
                            for substation, action_converter in self.substation_to_action_converter.items()
                        },
                        **self.architecture.pop.dictatorship_penalties
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
        chosen_communities: List[Community],
        incentives: Optional[Dict[Manager, float]] = None,
        dictatorship_penalties: Optional[Dict[Manager, float]] = None,
    ) -> None:
        chosen_communities_to_manager = {
            community: manager
            for community, manager in self.community_to_manager.items()
            if community in chosen_communities
        }
        if done:
            losses, rewards = zip(
                *ray.get(
                    [
                        manager.step.remote(
                            observation=community_to_sub_graphs_dict[old_community],
                            action=actions[old_community],
                            reward=(
                                reward + incentives[manager]
                                if incentives is not None
                                else reward
                            ),
                            next_observation=dgl.DGLGraph(),
                            done=done,
                            stop_decay=stop_decay[old_community],
                        )
                        for old_community, manager in chosen_communities_to_manager.items()
                    ]
                )
            )
        else:
            # Build a mapping between each manager and the communities she handles
            # The mapping represents communities before and after applying the chosen action
            manager_to_community_transformation: Dict[
                Manager, Tuple[List[Community], List[Community]]
            ] = {manager: ([], []) for manager in self.managers_history.keys()}
            for community, manager in chosen_communities_to_manager.items():
                manager_to_community_transformation[manager][0].append(community)

            for new_community, manager in next_community_to_manager.items():
                manager_to_community_transformation[manager][1].append(new_community)

            # Step the managers
            # For each community before applying the action
            # Find the most similar community among the ones managed after applying the action
            try:
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
                                            reward=(
                                                reward + incentives[manager]
                                                if incentives is not None
                                                else reward
                                            )
                                            + (
                                                0
                                                if not dictatorship_penalties
                                                else dictatorship_penalties[manager]
                                            ),
                                            next_observation=next_community_to_sub_graphs_dict[
                                                new_manager_communities[
                                                    np.argmax(
                                                        [
                                                            self._jaccard_distance(
                                                                old_community,
                                                                new_community,
                                                            )
                                                            for new_community in new_manager_communities
                                                        ]
                                                    )
                                                ]
                                            ]
                                            if new_manager_communities
                                            else dgl.DGLGraph(),
                                            done=done,
                                            stop_decay=stop_decay[old_community],
                                        )
                                        for old_community in old_manager_communities
                                    ]
                                    for manager, (
                                        old_manager_communities,
                                        new_manager_communities,
                                    ) in manager_to_community_transformation.items()
                                ]
                            )
                        )
                    )
                )
            except ValueError as e:
                print("....")
                raise e

        names = ray.get(
            [
                chosen_communities_to_manager[community].get_name.remote()
                for community in chosen_communities
            ]
        )

        self.log_step(
            losses=losses,
            implicit_rewards=[full_reward - reward for full_reward in rewards],
            names=names,
            train_steps=self.train_steps,
            incentives=[
                incentives[chosen_communities_to_manager[community]]
                for community in chosen_communities
            ]
            if incentives is not None
            else None,
            dictatorship_penalties=[
                dictatorship_penalties[chosen_communities_to_manager[community]]
                for community in chosen_communities
            ]
            if dictatorship_penalties is not None
            else None,
        )

    def _step_agents(
        self,
        factored_observation: Dict[Substation, dgl.DGLHeteroGraph],
        actions: Dict[Substation, int],
        reward: float,
        next_factored_observation: Dict[Substation, dgl.DGLHeteroGraph],
        done: bool,
        stop_decay: Dict[Substation, bool],
        selected_substations: List[Substation],
        incentives: Optional[Dict[Substation, float]] = None,
    ) -> None:
        # List of promises to get with ray.get
        step_promises = []

        # Substations present in next_factored_observation
        substations = []

        subs_to_agent_to_step = {
            sub: agent
            for sub, agent in self.substation_to_agent.items()
            if sub in selected_substations
        }

        if done:
            for sub_id, observation in factored_observation.items():
                if sub_id in subs_to_agent_to_step.keys():
                    step_promises.append(
                        subs_to_agent_to_step[sub_id].step.remote(
                            observation=observation,
                            action=actions[sub_id],
                            reward=reward + incentives[sub_id]
                            if incentives is not None
                            else reward,
                            next_observation=dgl.DGLGraph(),
                            done=done,
                            stop_decay=stop_decay[sub_id],
                        )
                    )
        else:
            for sub_id, observation in factored_observation.items():
                next_observation: Optional[
                    dgl.DGLHeteroGraph
                ] = next_factored_observation.get(sub_id)
                if (
                    next_observation is not None
                    and sub_id in subs_to_agent_to_step.keys()
                ):

                    # Build step promise with the previous and next observation
                    step_promises.append(
                        subs_to_agent_to_step[sub_id].step.remote(
                            observation=observation,
                            action=actions[sub_id],
                            reward=reward + incentives[sub_id]
                            if incentives is not None
                            else reward,
                            next_observation=next_factored_observation[sub_id],
                            done=done,
                            stop_decay=stop_decay[sub_id],
                        )
                    )
                    substations.append(sub_id)

        # Step the agents
        losses, rewards = zip(*ray.get(step_promises))
        names = ray.get(
            [
                subs_to_agent_to_step[substation].get_name.remote()
                for substation in substations
            ]
        )
        self.log_step(
            losses=losses,
            implicit_rewards=[full_reward - reward for full_reward in rewards],
            names=names,
            train_steps=self.train_steps,
            incentives=[incentives[substation] for substation in substations]
            if incentives is not None
            else None,
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
            node_embeddings = ray.get(
                current_community_to_manager_dict[community].get_node_embeddings.remote(
                    sub_graph
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
            summarized_graph.nodes[community_list[0]][
                self.architecture.pop.head_manager_embedding_name
            ] = th.cat(
                (
                    th.sigmoid(summarized_graph.nodes[community_list[0]]["embedding"]),
                    th.tensor(
                        [
                            1 if substation in community else 0
                            for substation in range(self.env.n_sub * 2)
                        ]
                    ).to(self.device),
                    th.tensor([manager_actions[community]]).to(self.device),
                ),
                dim=-1,
            ).squeeze()

        # The summarized graph is returned in DGL format
        # Each supernode has the action chosen by its community manager
        # And the contracted embedding
        return dgl.from_networkx(
            summarized_graph.to_directed(),
            node_attrs=[self.architecture.pop.head_manager_embedding_name],
            device=self.device,
        )


def train(
    env: BaseEnv,
    iterations: int,
    dpop: BasePOP,
    save_frequency: int = 3600,
):

    training_step: int = 0
    obs: BaseObservation = env.reset()
    done = False
    reward = env.reward_range[0]
    total_episodes = len(env.chronics_handler.subpaths)

    last_save_time = time.time()
    env.chronics_handler.shuffle()
    print("Model will be checkpointed every " + str(save_frequency) + " seconds")
    with tqdm(total=iterations - training_step) as pbar:
        while training_step < iterations:
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

            if dpop.episodes % total_episodes == 0:
                env.chronics_handler.shuffle()
            if done:
                obs = env.reset()
                dpop.writer.flush()

            pbar.update(1)

    print("\nSaving\n")
    dpop.writer.close()
    dpop.save()
    ray.shutdown()
