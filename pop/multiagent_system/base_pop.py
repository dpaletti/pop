from abc import abstractmethod
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
        self.node_number: Optional[int] = None

        # Training or Evaluation
        self.training = training

        # Logging
        self.train_steps: int = 0
        self.episodes: int = 0
        self.alive_steps: int = 0

        # State needed for step()
        # This state is saved so that after my_act()
        # step() can use pre-computed values
        self.chosen_action: Optional[EncodedAction] = None
        self.manager_actions: Optional[Dict[Community, EncodedAction]] = None
        self.agent_actions: Optional[Dict[Substation, EncodedAction]] = None
        self.sub_graphs: Optional[Dict[Community, dgl.DGLHeteroGraph]] = None
        self.summarized_graph: Optional[dgl.DGLHeteroGraph] = None
        self.factored_observation: Optional[Dict[Substation, dgl.DGLHeteroGraph]] = None

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
        self, first_observation: BaseObservation, first_observation_graph: nx.Graph
    ) -> None:
        self.node_number = len(first_observation_graph.nodes)

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
            first_observation, first_observation_graph, self.converter
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
                agent_actions=first_observation_graph.nodes,
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
        self, factored_observation: Dict[Substation, dgl.DGLHeteroGraph]
    ) -> Dict[Substation, EncodedAction]:
        action_list: List[int] = ray.get(
            [
                self.sub_to_agent_dict[sub_id].take_action.remote(
                    transformed_observation=observation
                )
                for sub_id, observation in factored_observation.items()
            ]
        )
        return {
            sub_id: self.lookup_local_action(
                self.sub_to_agent_converters_dict[sub_id].all_actions[action]
            )
            for sub_id, action in zip(factored_observation.keys(), action_list)
        }

    def get_manager_actions(
        self, community_to_subgraphs_dict: Dict[Community, dgl.DGLHeteroGraph]
    ) -> Dict[Community, EncodedAction]:
        action_list: List[Substation] = ray.get(
            [
                self.community_to_manager_dict[community].take_action.remote(
                    transformed_observation=community_to_subgraphs_dict[community],
                    mask=community,
                    return_embedding=True,
                )
                for community in self.communities
            ]
        )
        return {
            community: self.agent_actions[action]
            for community, action in zip(self.communities, action_list)
        }

    @abstractmethod
    def get_action(self, observation: dgl.DGLHeteroGraph) -> EncodedAction:
        ...

    def step_managers(
        self,
        community_to_sub_graphs_dict: Dict[Community, dgl.DGLHeteroGraph],
        action: EncodedAction,
        reward: float,
        next_community_to_sub_graphs_dict: Dict[Community, dgl.DGLHeteroGraph],
        done: bool,
        stop_decay: Dict[Community, bool],
    ) -> None:
        ray.get(
            [
                self.community_to_manager_dict[community].step.remote(
                    observation=community_to_sub_graphs_dict[community],
                    action=action,
                    reward=reward,
                    next_observation=next_community_to_sub_graphs_dict[community],
                    done=done,
                    stop_decay=stop_decay[community],
                )
                for community in self.communities
            ]
        )

    def step_agents(
        self,
        factored_observation: Dict[Substation, dgl.DGLHeteroGraph],
        action: EncodedAction,
        reward: float,
        next_factored_observation: Dict[Substation, dgl.DGLHeteroGraph],
        done: bool,
        stop_decay: Dict[Substation, bool],
    ) -> None:
        ray.get(
            [
                self.sub_to_agent_dict[sub_id].step.remote(
                    observation=observation,
                    action=action,
                    reward=reward,
                    next_observation=next_factored_observation[sub_id],
                    done=done,
                    stop_decay=stop_decay[sub_id],
                )
                for sub_id, observation in factored_observation.items()
            ]
        )

    @abstractmethod
    def learn(self, reward: float):
        ...

    def my_act(
        self,
        transformed_observation: Tuple[Dict[Substation, dgl.DGLHeteroGraph], nx.Graph],
        reward: float,
        done=False,
    ) -> EncodedAction:
        graph: nx.Graph

        self.factored_observation, graph = transformed_observation

        # Observation as a neighbourhood for each node

        if self.communities and not self.fixed_communities:
            raise Exception("\nDynamic Communities are not implemented yet\n")

        # Actions retreived are encoded with local agent converters
        # need to be remapped to the global converter
        self.agent_actions: Dict[Substation, EncodedAction] = self.get_agent_actions(
            self.factored_observation
        )

        # Each agent is assigned to its chosen action
        nx.set_node_attributes(
            graph,
            {
                node_id: {
                    "action": self.agent_actions[node_data["sub_id"]],
                }
                for node_id, node_data in graph.nodes.data()
            },
        )

        # The main graph is split into communities
        self.sub_graphs: Dict[
            Community, dgl.DGLHeteroGraph
        ] = split_graph_into_communities(graph, self.communities, str(self.device))

        # Managers chooses the best action
        self.manager_actions: Dict[Community, EncodedAction] = self.get_manager_actions(
            self.sub_graphs
        )

        # The graph is summarized by contracting every community in 1 supernode
        # And storing the embedding of each manager in each supernode as node feature
        # Together with the action chosen by the manager
        self.summarized_graph: dgl.DGLHeteroGraph = self.summarize_graph(
            graph, self.manager_actions, self.sub_graphs
        ).to(self.device)

        # The head manager chooses the best action from every community given the summarized graph
        self.chosen_action: EncodedAction = self.get_action(self.summarized_graph)

        # Log to Tensorboard
        self.log_system_behaviour(
            best_action=self.chosen_action,
            manager_actions={
                self.community_to_manager_dict[community].name.split("_")[1]: action
                for community, action in self.manager_actions.items()
            },
            agent_actions={
                self.sub_to_agent_dict[sub_id].name.split("_")[1]: action
                for sub_id, action in self.agent_actions.items()
            },
            train_steps=self.train_steps,
        )

        return self.chosen_action

    def _extra_step(
        self,
        action: EncodedAction,
        reward: float,
        next_sub_graphs: Dict[Community, dgl.DGLHeteroGraph],
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
            self.episodes += 1
            self.alive_steps = 0
            self.train_steps += 1
        else:

            next_graph: nx.Graph = next_observation.as_networkx()
            next_factored_observation: Dict[
                Substation, dgl.DGLHeteroGraph
            ] = factor_observation(
                next_graph,
                device=str(self.device),
                radius=self.architecture.pop.agent_neighbourhood_radius,
            )

            agents_stop_decay: Dict[Substation, bool] = {
                sub_id: True if action == self.chosen_action else False
                for sub_id, action in self.agent_actions.items()
            }

            next_agent_actions: Dict[
                Substation, EncodedAction
            ] = self.get_agent_actions(next_factored_observation)

            nx.set_node_attributes(
                next_graph,
                {
                    node_id: {
                        "action": next_agent_actions[node_data["sub_id"]],
                    }
                    for node_id, node_data in next_graph.nodes.data()
                },
            )

            if not self.architecture.pop.fixed_communities:
                # TODO recompute community clustering for observation and next_observation
                raise Exception("Dynamic communities not yet implemented.")

            next_sub_graphs: Dict[
                Community, dgl.DGLHeteroGraph
            ] = split_graph_into_communities(
                next_graph, self.communities, str(self.device)
            )

            manager_stop_decay: Dict[Community, bool] = {
                community: True if action == self.chosen_action else False
                for community, action in self.manager_actions
            }

            self._extra_step(
                next_sub_graphs=next_sub_graphs,
                next_graph=next_graph,
                reward=reward,
                action=action,
                done=done,
            )

            self.step_managers(
                community_to_sub_graphs_dict=self.sub_graphs,
                action=action,
                reward=reward,
                next_community_to_sub_graphs_dict=next_sub_graphs,
                done=done,
                stop_decay=manager_stop_decay,
            )

            self.step_agents(
                factored_observation=self.factored_observation,
                action=action,
                reward=reward,
                next_factored_observation=next_factored_observation,
                done=done,
                stop_decay=agents_stop_decay,
            )

            self.train_steps += 1
            self.alive_steps += 1

    def convert_obs(
        self, observation: BaseObservation
    ) -> Tuple[Dict[Substation, dgl.DGLHeteroGraph], nx.Graph]:

        observation_graph: nx.Graph = observation.as_networkx()
        factored_observation = factor_observation(
            observation_graph,
            str(self.device),
            self.architecture.pop.agent_neighbourhood_radius,
        )

        if self.node_features is None:
            # if this is the first observation received by the agent system
            self.finalize_init_on_first_observation(observation, observation_graph)

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
            sub_graphs[community].ndata[
                "node_embeddings"
            ] = self.community_to_manager_dict[community].get_node_embeddings()
            for node in community:
                node_attribute_dict[node] = {
                    "embedding": dgl.mean_nodes(
                        sub_graphs[community], "node_embeddings"
                    )
                }
                node_attribute_dict[node] = {"community": community}

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
            summarized_graph.nodes[community_list[0]]["action"] = manager_actions[
                community
            ]

        # The summarized graph is returned in DGL format
        # Each supernode has the action chosen by its community manager
        # And the contracted embedding
        return dgl.from_networkx(
            summarized_graph.to_directed(),
            node_attrs=["action", "embedding", "community"],
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
        }

    # TODO: finish serialization implementation


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
                action=action,
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
