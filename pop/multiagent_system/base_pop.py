from abc import abstractmethod
from dataclasses import asdict
from typing import Optional, List, Tuple, Dict, Union, Any

import dgl
import networkx as nx
from grid2op.Action import BaseAction
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv
import torch as th
from grid2op.Observation import BaseObservation

from random import choice

from tqdm import tqdm

from agents.manager import Manager
from agents.ray_gcn_agent import RayGCNAgent
from agents.ray_shallow_gcn_agent import RayShallowGCNAgent
from community_detection.community_detector import CommunityDetector, Community
from configs.architecture import Architecture
from multiagent_system.space_factorization import (
    factor_action_space,
    HashableAction,
    factor_observation,
    split_graph_into_communities,
    Substation,
    EncodedAction,
)
from networks.serializable_module import SerializableModule
from agents.loggable_module import LoggableModule
import ray


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
        pre_initialized: bool = False,
    ):
        AgentWithConverter.__init__(self, env.action_space, IdToAct)
        SerializableModule.__init__(self, checkpoint_dir, name)
        LoggableModule.__init__(self, tensorboard_dir)

        self.name = name
        self.seed = seed
        self.env = env

        # Converter
        self.converter = IdToAct(env.action_space)
        self.converter.init_converter()
        self.converter.seed(seed)

        # Setting the device
        if device is None:
            self.device: th.device = th.device(
                "cuda:0" if th.cuda.is_available() else "cpu"
            )
        else:
            self.device: th.device = th.device(device)

        self.architecture: Architecture = architecture

        # Node and edge features
        self.env = env
        self.node_features: Optional[int] = None
        self.edge_features: Optional[int] = None

        # Training or Evaluation
        self.pre_initialized = pre_initialized  # skips part of post_init
        self.initialized = False
        self.training = training

        # Logging
        self.train_steps: int = 0
        self.episodes: int = 0
        self.alive_steps: int = 0

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
        self.substation_to_encoded_action: Optional[Substation, EncodedAction] = None

        # Agents
        self.action_lookup_table: Optional[Dict[HashableAction, int]] = None
        self.sub_to_agent_converters_dict: Optional[Dict[Substation, IdToAct]] = None
        self.sub_to_agent_dict: Optional[
            Dict[Substation, Union[RayGCNAgent, RayShallowGCNAgent]]
        ] = None

        # Managers
        self.community_to_manager_dict: Optional[Dict[Community, Manager]] = None

        # Community Detector Initialization
        self.community_detector = CommunityDetector(seed)
        self.fixed_communities = self.architecture.pop.fixed_communities
        if not self.fixed_communities:
            # TODO: implement dynamic communities
            # TODO: each time a new observation is evaluated (both in act() and step())
            # TODO: communities must be re-computed and Manager dict must be updated consistently
            raise Exception("Dynamic communities not yet implemented")
        self.communities: Optional[List[Community]] = None

    def finalize_init_on_first_observation(
        self,
        first_observation: BaseObservation,
        first_observation_graph: nx.Graph,
        pre_initialized=False,
    ) -> None:
        if pre_initialized:
            sub_id_to_action_space, self.action_lookup_table = factor_action_space(
                first_observation,
                first_observation_graph,
                self.converter,
                self.env.n_sub,
            )
            self.sub_to_agent_converters_dict: Dict[int, IdToAct] = {}
            for sub_id, action_space in sub_id_to_action_space.items():
                conv = IdToAct(self.env.action_space)
                conv.init_converter(action_space)
                conv.seed(self.seed)
                self.sub_to_agent_converters_dict[sub_id] = conv

            sub_id_to_action_space, self.action_lookup_table = factor_action_space(
                first_observation,
                first_observation_graph,
                self.converter,
                self.env.n_sub,
            )

            self.log_action_space_size(
                agent_converters=self.sub_to_agent_converters_dict
            )
            return

        # Compute node and edge features
        self.node_features = len(
            first_observation_graph.nodes[
                choice(list(first_observation_graph.nodes))
            ].keys()
        )
        self.edge_features = len(
            first_observation_graph.edges[
                choice(list(first_observation_graph.edges))
            ].keys()
        )

        # Compute first community structure
        self.communities = self.community_detector.dynamo(
            graph_t=first_observation_graph
        )

        # Agents Initialization
        sub_id_to_action_space, self.action_lookup_table = factor_action_space(
            first_observation, first_observation_graph, self.converter, self.env.n_sub
        )

        # Agents Initialization
        self.sub_to_agent_dict: Dict[int, Union[RayGCNAgent, RayShallowGCNAgent]] = {
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
            for sub_id, action_space in sub_id_to_action_space.items()
        }

        self.sub_to_agent_converters_dict: Dict[int, IdToAct] = {}
        for sub_id, action_space in sub_id_to_action_space.items():
            conv = IdToAct(self.env.action_space)
            conv.init_converter(action_space)
            conv.seed(self.seed)
            self.sub_to_agent_converters_dict[sub_id] = conv

        self.log_action_space_size(agent_converters=self.sub_to_agent_converters_dict)

        # Managers Initialization
        self.community_to_manager_dict: Dict[Community, Manager] = {
            community: Manager.remote(
                agent_actions=len(first_observation_graph.nodes),
                node_features=self.node_features + 1,  # Node Features + Action
                edge_features=self.edge_features,
                architecture=self.architecture.manager,
                name="manager_" + str(idx) + "_" + self.name,
                training=self.training,
                device=self.device,
            )
            for idx, community in enumerate(self.communities)
        }

    def get_agent_actions(
        self, factored_observation: Dict[Substation, Optional[dgl.DGLHeteroGraph]]
    ) -> Dict[Substation, int]:
        action_list: list = [
            self.sub_to_agent_dict[sub_id].take_action.remote(
                transformed_observation=observation
            )
            if observation is not None
            else 0
            for sub_id, observation in factored_observation.items()
        ]
        actions_accomplished: List[int] = ray.get(
            [action for action in action_list if action != 0]
        )
        action_list = [
            0 if action == 0 else actions_accomplished[idx]
            for idx, action in enumerate(action_list)
        ]
        return {
            sub_id: action
            for sub_id, action in zip(factored_observation.keys(), action_list)
        }

    def get_manager_actions(
        self, community_to_sub_graphs_dict: Dict[Community, dgl.DGLHeteroGraph]
    ) -> Dict[Community, Substation]:
        action_list: List[Substation] = ray.get(
            [
                self.community_to_manager_dict[community].take_action.remote(
                    transformed_observation=community_to_sub_graphs_dict[community],
                    mask=frozenset(
                        [
                            sub
                            for sub in community
                            if sub in list(self.substation_to_encoded_action.keys())
                        ]
                    ),
                )
                for community in self.communities
            ]
        )
        return {
            community: action
            for community, action in zip(self.communities, action_list)
        }

    @abstractmethod
    def get_action(self, observation: dgl.DGLHeteroGraph) -> int:
        ...

    def step_managers(
        self,
        community_to_sub_graphs_dict: Dict[Community, dgl.DGLHeteroGraph],
        actions: Dict[Community, Substation],
        reward: float,
        next_community_to_sub_graphs_dict: Dict[Community, dgl.DGLHeteroGraph],
        done: bool,
        stop_decay: Dict[Community, bool],
    ) -> None:
        losses = ray.get(
            [
                self.community_to_manager_dict[community].step.remote(
                    observation=community_to_sub_graphs_dict[community],
                    action=actions[community],
                    reward=reward,
                    next_observation=next_community_to_sub_graphs_dict[community],
                    done=done,
                    stop_decay=stop_decay[community],
                )
                for community in self.communities
            ]
        )

        self.log_loss(
            {
                manager_name: loss
                for manager_name, loss in zip(
                    ray.get(
                        [
                            self.community_to_manager_dict[community].get_name.remote()
                            for community in self.communities
                        ]
                    ),
                    losses,
                )
                if loss is not None
            },
            self.train_steps,
        )

    def step_agents(
        self,
        factored_observation: Dict[Substation, dgl.DGLHeteroGraph],
        actions: Dict[Substation, int],
        reward: float,
        next_factored_observation: Dict[Substation, dgl.DGLHeteroGraph],
        done: bool,
        stop_decay: Dict[Substation, bool],
    ) -> None:
        step_promises = []
        substations = []
        for sub_id, observation in factored_observation.items():
            next_observation: Optional[
                dgl.DGLHeteroGraph
            ] = next_factored_observation.get(sub_id)
            if next_observation is not None:
                step_promises.append(
                    self.sub_to_agent_dict[sub_id].step.remote(
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
                    "Substation '"
                    + str(sub_id)
                    + "' not present in current next_state, related agent is not stepping..."
                )
        losses = ray.get(step_promises)
        self.log_loss(
            {
                agent_name: loss
                for agent_name, loss in zip(
                    ray.get(
                        [
                            self.sub_to_agent_dict[substation].get_name.remote()
                            for substation in substations
                        ]
                    ),
                    losses,
                )
                if loss is not None
            },
            self.train_steps,
        )

    def _compute_managers_sub_graphs(
        self, graph: nx.graph, substation_to_local_action: Dict[Substation, int]
    ) -> Tuple[Dict[Community, dgl.DGLHeteroGraph], Dict[Substation, EncodedAction]]:
        substation_to_encoded_action: Dict[Substation, EncodedAction] = {
            sub_id: self.lookup_local_action(
                self.sub_to_agent_converters_dict[sub_id].all_actions[local_action]
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
            split_graph_into_communities(graph, self.communities, str(self.device)),
            substation_to_encoded_action,
        )

    def _compute_summarized_graph(
        self,
        graph,
        sub_graphs: Dict[Community, dgl.DGLHeteroGraph],
        substation_to_encoded_action: Dict[Substation, EncodedAction],
        community_to_substation: Dict[Community, Substation],
    ):
        # The graph is summarized by contracting every community in 1 supernode
        # And storing the embedding of each manager in each supernode as node feature
        # Together with the action chosen by the manager
        return self.summarize_graph(
            graph,
            {
                community: substation_to_encoded_action[substation]
                for community, substation in community_to_substation.items()
            },
            sub_graphs,
        ).to(self.device)

    def my_act(
        self,
        transformed_observation: Tuple[
            Dict[Substation, Optional[dgl.DGLHeteroGraph]], nx.Graph
        ],
        reward: float,
        done=False,
    ) -> EncodedAction:
        graph: nx.Graph

        self.factored_observation, graph = transformed_observation

        # Observation as a neighbourhood for each node

        if self.communities and not self.fixed_communities:
            raise Exception("\nDynamic Communities are not implemented yet\n")

        # Actions retrieved are encoded with local agent converters
        # need to be remapped to the global converter
        self.substation_to_local_action: Dict[Substation, int] = self.get_agent_actions(
            self.factored_observation
        )

        # Split graph into communities
        (
            self.sub_graphs,
            self.substation_to_encoded_action,
        ) = self._compute_managers_sub_graphs(
            graph=graph, substation_to_local_action=self.substation_to_local_action
        )

        # Managers chooses the best substation
        self.community_to_substation: Dict[
            Community, Substation
        ] = self.get_manager_actions(self.sub_graphs)

        self.summarized_graph = self._compute_summarized_graph(
            graph=graph,
            sub_graphs=self.sub_graphs,
            substation_to_encoded_action=self.substation_to_encoded_action,
            community_to_substation=self.community_to_substation,
        )

        # The head manager chooses the best action from every community given the summarized graph
        self.chosen_node = self.get_action(self.summarized_graph)

        self.chosen_community = frozenset(
            [
                substation
                for substation, belongs_to_community in enumerate(
                    self.summarized_graph.ndata["embedding_action"][self.chosen_node][
                        -(self.env.n_sub + 1) : -1
                    ]
                    .detach()
                    .tolist()
                )
                if belongs_to_community
            ]
        )

        self.chosen_action = self.substation_to_encoded_action[
            self.community_to_substation[self.chosen_community]
        ]

        # Log to Tensorboard
        self.log_system_behaviour(
            best_action=self.chosen_action,
            manager_actions={
                ray.get(
                    self.community_to_manager_dict[community].get_name.remote()
                ).split("_")[1]: action
                for community, action in self.community_to_substation.items()
            },
            agent_actions={
                ray.get(self.sub_to_agent_dict[sub_id].get_name.remote()).split("_")[
                    1
                ]: action
                for sub_id, action in self.substation_to_local_action.items()
            },
            train_steps=self.train_steps,
        )

        return self.chosen_action

    def _extra_step(
        self,
        action: int,
        reward: float,
        next_sub_graphs: Dict[Community, dgl.DGLHeteroGraph],
        next_substation_to_encoded_action: Dict[Substation, EncodedAction],
        next_graph: nx.Graph,
        done: bool,
    ):
        # Override this to extend step()
        pass

    def step(
        self,
        action: EncodedAction,
        observation: BaseObservation,  # kept for interface consistency
        reward: float,
        next_observation: BaseObservation,
        done: bool,
    ):
        if done:
            self.log_alive_steps(self.alive_steps, self.episodes)
            self.train_steps += 1
            self.episodes += 1
            self.alive_steps = 0
        else:
            self.log_reward(reward, self.train_steps)
            self.alive_steps += 1
            self.train_steps += 1

            next_graph: nx.Graph = next_observation.as_networkx()
            next_factored_observation: Dict[
                Substation, Optional[dgl.DGLHeteroGraph]
            ] = factor_observation(
                next_graph,
                device=str(self.device),
                radius=self.architecture.pop.agent_neighbourhood_radius,
            )

            # WARNING: for no-action there may be multiple agents playing such action
            # WARNING: this is intended to encourage sparse policies
            agents_stop_decay: Dict[Substation, bool] = {
                sub_id: False if agent_action == self.chosen_action else True
                for sub_id, agent_action in self.substation_to_encoded_action.items()
            }

            next_substation_to_local_actions: Dict[
                Substation, int
            ] = self.get_agent_actions(next_factored_observation)

            (
                next_sub_graphs,
                next_substation_to_encoded_action,
            ) = self._compute_managers_sub_graphs(
                graph=next_graph,
                substation_to_local_action=next_substation_to_local_actions,
            )

            if not self.architecture.pop.fixed_communities:
                # TODO recompute community clustering for observation and next_observation
                raise Exception("Dynamic communities not yet implemented.")

            manager_stop_decay: Dict[Community, bool] = {
                community: False if community == self.chosen_community else True
                for community, _ in self.community_to_manager_dict.items()
            }

            self._extra_step(
                next_sub_graphs=next_sub_graphs,
                next_graph=next_graph,
                next_substation_to_encoded_action=next_substation_to_encoded_action,
                reward=reward,
                action=self.chosen_node,
                done=done,
            )

            self.step_managers(
                community_to_sub_graphs_dict=self.sub_graphs,
                actions=self.community_to_substation,
                reward=reward,
                next_community_to_sub_graphs_dict=next_sub_graphs,
                done=done,
                stop_decay=manager_stop_decay,
            )

            self.step_agents(
                factored_observation=self.factored_observation,
                actions=self.substation_to_local_action,
                reward=reward,
                next_factored_observation=next_factored_observation,
                done=done,
                stop_decay=agents_stop_decay,
            )

    def convert_obs(
        self, observation: BaseObservation
    ) -> Tuple[Dict[Substation, Optional[dgl.DGLHeteroGraph]], nx.Graph]:

        observation_graph: nx.Graph = observation.as_networkx()
        factored_observation: Dict[
            Substation, Optional[dgl.DGLHeteroGraph]
        ] = factor_observation(
            observation_graph,
            str(self.device),
            self.architecture.pop.agent_neighbourhood_radius,
        )

        if not self.initialized:
            # if this is the first observation received by the agent system

            self.finalize_init_on_first_observation(
                observation, observation_graph, pre_initialized=self.pre_initialized
            )
            self.initialized = True

        return factored_observation, observation_graph

    def lookup_local_action(self, action: BaseAction):
        return self.action_lookup_table[HashableAction(action)]

    def summarize_graph(
        self,
        graph: nx.Graph,
        manager_actions: Dict[Community, EncodedAction],
        sub_graphs: Dict[Community, dgl.DGLHeteroGraph],
    ) -> dgl.DGLHeteroGraph:

        # CommunityManager node embedding is assigned to each node
        node_attribute_dict = {}
        for community, sub_graph in sub_graphs.items():
            node_embeddings = ray.get(
                self.community_to_manager_dict[community].get_node_embeddings.remote()
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
        for community in self.communities:
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

    def get_state(self: "BasePOP") -> Dict[str, Any]:
        agents_state_list: List[Dict[str, Any]] = ray.get(
            [agent.get_state.remote() for _, agent in self.sub_to_agent_dict.items()]
        )
        managers_state_list: List[Dict[str, Any]] = ray.get(
            [
                manager.get_state.remote()
                for _, manager in self.community_to_manager_dict.items()
            ]
        )
        return {
            "agents_state": {
                sub_id: agent_state
                for sub_id, agent_state in zip(
                    list(self.sub_to_agent_dict.keys()), agents_state_list
                )
            },
            "managers_state": {
                community: manager_state
                for community, manager_state in zip(
                    list(self.community_to_manager_dict.keys()), managers_state_list
                )
            },
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "train_steps": self.train_steps,
            "episodes": self.episodes,
            "alive_steps": self.alive_steps,
            "name": self.name,
            "architecture": asdict(self.architecture),
            "seed": self.seed,
            "device": str(self.device),
            "communities": self.communities,
        }


def train(env: BaseEnv, iterations: int, dpop):

    training_step: int = 0
    obs: BaseObservation = (
        env.reset()
    )  # Typing issue for env.reset(), returns BaseObservation
    done = False
    reward = env.reward_range[0]
    total_episodes = len(env.chronics_handler.subpaths)
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
            pbar.update(1)

    print("\nSaving...\n")

    dpop.save()
