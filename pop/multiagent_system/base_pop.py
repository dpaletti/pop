from abc import abstractmethod
from typing import Union, Optional, List, Set, Tuple, Dict

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

from community_detection.community_detector import CommunityDetector
from configs.architecture import Architecture
from multiagent_system.space_factorization import (
    factor_action_space,
    HashableAction,
    factor_observation,
    split_graph_into_communities,
)
from networks.serializable_module import SerializableModule
from agents.loggable_module import LoggableModule


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

        # Compute node and edge features
        self.graph = env.reset().as_networkx()
        self.node_features = len(
            self.graph.nodes[choice(list(self.graph.nodes))].keys()
        )
        self.edge_features = len(
            self.graph.edges[choice(list(self.graph.edges))].keys()
        )

        # Training or Evaluation
        self.training = training

        # Logging
        self.train_steps: int = 0
        self.episodes: int = 0
        self.alive_steps: int = 0
        self.managers_learning_steps: int = 0
        self.agent_learning_steps: int = 0
        self.current_chosen_node: int = -1
        self.current_chosen_manager: int = -1
        self.local_actions = []

        # Agents Initialization
        self.action_spaces, self.action_lookup_table = factor_action_space(env)
        self.factored_observation = []

        self.agent_chosen_actions: List[List[int]] = [
            [] for _ in range(len(self.action_spaces))
        ]

        self.agent_converters: List[IdToAct] = []
        for action_space in self.action_spaces:
            conv = IdToAct(env.action_space)
            conv.init_converter(action_space)
            conv.seed(seed)
            self.agent_converters.append(conv)

        self.log_action_space_size(agent_converters=self.agent_converters)

        # Community Detector Initialization
        self.community_detector = CommunityDetector(seed)
        self.fixed_communities = self.architecture.pop.fixed_communities
        self.communities: List[Tuple[int, ...]] = [
            tuple(community) for community in self.initialize_communities()
        ]
        if not self.fixed_communities:
            raise Exception("\nDynamic Communities are not implemented yet\n")

    @property
    @abstractmethod
    def agents(self):
        ...

    @property
    @abstractmethod
    def managers(self):
        ...

    @abstractmethod
    def get_agent_actions(self, factored_observation):
        ...

    @abstractmethod
    def get_manager_actions(self, graph: Dict[Tuple[int, ...], dgl.DGLHeteroGraph]):
        ...

    @abstractmethod
    def get_action(self, *args) -> int:
        ...

    @abstractmethod
    def teach_managers(self, manager_losses):
        ...

    @abstractmethod
    def learn(self, reward: float):
        ...

    @abstractmethod
    def step_agents(self, next_observation, reward, done):
        ...

    def my_act(
        self,
        transformed_observation: Tuple[List[dgl.DGLHeteroGraph], nx.Graph],
        reward,
        done=False,
    ):

        graph: nx.Graph = transformed_observation[1]

        self.factored_observation: List[dgl.DGLHeteroGraph] = transformed_observation[0]

        if self.communities and not self.fixed_communities:
            raise Exception("\nDynamic Communities are not implemented yet\n")

        (
            self.local_actions,
            encoded_local_actions,
            current_epsilons,
        ) = self.get_agent_actions(self.factored_observation)

        # Each agent is assigned to its chosen (global) action
        nx.set_node_attributes(
            graph,
            {
                node: {"action": self.lookup_local_action(action), "global_id": node}
                for node, action in zip(graph.nodes, self.local_actions)
            },
        )

        # The main graph is split into communities
        sub_graphs = split_graph_into_communities(graph, self.communities, self.device)

        # Managers choose the best node given the action it chose
        chosen_nodes, manager_epsilons = self.get_manager_actions(sub_graphs)
        managed_actions = [
            graph.nodes[chosen_node]["action"] for chosen_node in chosen_nodes
        ]

        # The graph is summarized by contracting every community in 1 supernode
        summarized_graph: dgl.DGLHeteroGraph = self.summarize_graph(
            graph, sub_graphs, managed_actions
        ).to(self.device)

        # The head manager chooses the best action from every community given the summarized graph
        best_action, self.current_chosen_manager = self.get_action(summarized_graph)

        # Global ID of the current chosen node
        self.current_chosen_node = (
            sub_graphs[self.current_chosen_manager]
            .ndata["global_id"][chosen_nodes[self.current_chosen_manager]]
            .item()
        )

        self.log_system_behaviour(
            best_action=best_action,
            manager_actions=managed_actions,
            agent_actions=encoded_local_actions,
            best_node=self.current_chosen_node,
            manager_nodes=chosen_nodes,
            best_manager=self.current_chosen_manager,
            converter=self.converter,
            agent_converters=self.agent_converters,
            agent_epsilons=current_epsilons,
            train_steps=self.train_steps,
        )

        return best_action

    def step(
        self,
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
            (
                losses,
                q_network_states,
                target_network_states,
                _,
            ) = self.step_agents(next_observation, reward, done)
            if not (None in losses):
                self.log_agents_loss(losses, self.agent_learning_steps)

                # WARNING: this is an expensive call
                self.log_agents_embedding_histograms(
                    q_network_states, target_network_states, self.agent_learning_steps
                )
                self.agent_learning_steps += 1

            self.train_steps += 1
            self.alive_steps += 1
            self.learn(reward)

    def convert_obs(
        self, observation: BaseObservation
    ) -> Tuple[List[dgl.DGLHeteroGraph], nx.Graph]:
        return factor_observation(
            observation, self.device, self.architecture.pop.agent_neighbourhood_radius
        )

    def initialize_communities(self) -> List[Set[int]]:
        obs = self.env.reset()
        graph = obs.as_networkx()
        communities = self.community_detector.dynamo(graph_t=graph)
        return communities

    def lookup_local_action(self, action: BaseAction):
        return self.action_lookup_table[HashableAction(action)]

    def summarize_graph(
        self,
        graph: nx.Graph,
        embedded_graphs: Dict[Tuple[int, ...], dgl.DGLHeteroGraph],
        managed_actions: List[int],
    ) -> dgl.DGLHeteroGraph:

        # CommunityManager node embedding is assigned to each node
        node_attribute_dict = {}
        for community, sub_graph in embedded_graphs.items():
            for idx, node in enumerate(sorted(list(community))):
                node_attribute_dict[node] = {
                    "embedding": sub_graph.nodes[idx].data["embedding"].detach()
                }
        nx.set_node_attributes(
            graph,
            node_attribute_dict,
        )

        # Graph is summarized by contracting communities into supernodes
        summarized_graph: nx.graph = graph
        for community in self.communities:
            community_list = list(community)
            for node in community_list[1:]:
                summarized_graph = nx.contracted_nodes(
                    summarized_graph, community_list[0], node
                )

        # Embedding gets contracted
        # Each supernode holds the mean of the embedding of the underlying nodes
        for node, node_data in summarized_graph.nodes.data():
            node_data["embedding"] = th.mean(
                th.stack(
                    [
                        node_data["contraction"][contracted_node]["embedding"]
                        for contracted_node in list(node_data["contraction"].keys())
                    ]
                ),
                dim=0,
            ).squeeze()
            for idx, community in enumerate(self.communities):
                if node in community:
                    node_data["action"] = th.Tensor(
                        ((node_data["embedding"].shape[0] - 1) * [0])
                        + [managed_actions[idx]]
                    )

        # The summarized graph is returned in DGL format
        # Each supernode has the action chosen by its community manager
        # And the contracted embedding
        return dgl.from_networkx(
            summarized_graph.to_directed(),
            node_attrs=["action", "embedding"],
            device=self.device,
        )


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
                reward=reward,
                next_observation=next_obs,
                done=done,
            )
            obs = next_obs
            training_step += 1
            pbar.update(1)

    print("\nSaving...\n")

    dpop.save()
