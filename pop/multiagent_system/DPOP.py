from typing import List, Optional, Set, Tuple
import json
import torch as th
import concurrent.futures

import dgl
import networkx as nx
from tqdm import tqdm
from joblib import Parallel, delayed
from grid2op.Agent import AgentWithConverter
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation

from pop.multiagent_system.space_factorization import (
    factor_action_space,
    factor_observation,
    HashableAction,
)
from node_agents.utilities import from_networkx_to_dgl
from node_agents.gcn_agent import DoubleDuelingGCNAgent
from community_detection.community_detector import CommunityDetector
from pop.multiagent_system.manager import Manager
import multiprocessing


class DPOP(AgentWithConverter):
    def __init__(
        self,
        env: BaseEnv,
        node_features: int,
        edge_features: int,
        fixed_communities: bool,
        name: str,
        architecture_path: str,
        training: bool,
        tensorboard_dir: Optional[str],
        checkpoint_dir: Optional[str],
        seed: int,
        device: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ):
        super(DPOP, self).__init__(env.action_space, IdToAct)

        self.architecture = json.load(open(architecture_path))

        # Environment Properties
        self.env = env
        self.node_features = node_features
        self.edge_features = edge_features

        # Checkpointing
        self.checkpoint_dir = checkpoint_dir

        if n_jobs is None:
            self.n_jobs = multiprocessing.cpu_count()
        else:
            self.n_jobs = n_jobs

        # Agents Initialization
        action_spaces, self.action_lookup_table = factor_action_space(env, self.n_jobs)
        self.agents: List[DoubleDuelingGCNAgent] = [
            DoubleDuelingGCNAgent(
                agent_actions=action_space,
                full_action_space=env.action_space,
                architecture=self.architecture["agent"],
                node_features=self.node_features,
                edge_features=self.edge_features,
                name="agent_" + str(idx) + "_" + name,
                seed=seed,
                training=training,
                tensorboard_log_dir=tensorboard_dir,
                log_dir=self.checkpoint_dir,
                device=device,
            )
            for idx, action_space in enumerate(action_spaces)
        ]

        # Community Detector Initialization
        self.community_detector = CommunityDetector(seed)
        self.fixed_communities = fixed_communities
        self.communities: List[Set[int]] = self.initialize_communities()
        if not self.fixed_communities:
            raise Exception("\nDynamic Communities are not implemented yet\n")

        # Managers Initialization
        self.managers: List[Manager] = [
            Manager(
                node_features=self.node_features + 1,  # Node Features + Action
                edge_features=self.edge_features,
                architecture=self.architecture["manager"],
                log_dir=self.checkpoint_dir,
                name="manager_" + str(idx) + "_" + name,
            )
            for idx, _ in enumerate(self.communities)
        ]
        if not self.architecture["decentralized"]:
            self.head_manager = Manager(
                node_features=2,  # Manager Embedding + Action
                edge_features=1,  # Edge Embedding
                architecture=self.architecture["head_manager"],
                name="head_manager_" + "_" + name,
                log_dir=self.checkpoint_dir,
            )

    def initialize_communities(self) -> List[Set[int]]:
        obs = self.env.reset()
        graph = self.get_graph(obs)
        communities = self.community_detector.dynamo(graph_t=graph)
        return communities

    def my_act(
        self,
        transformed_observation: Tuple[List[dgl.DGLHeteroGraph], nx.Graph],
        reward: float,
        done=False,
    ) -> int:
        graph: nx.Graph = transformed_observation[1]
        factored_observation = transformed_observation[0]
        if self.communities and not self.fixed_communities:
            raise Exception("\nDynamic Communities are not implemented yet\n")

        print("\nEach node_agents is computing its action:")
        local_actions = [
            agent.my_act(observation)
            for agent, observation in tqdm(zip(self.agents, factored_observation))
        ]

        # Action Encoding is converted from local to global
        global_actions = [
            self.action_lookup_table[
                HashableAction(agent.action_space_converter.all_actions[action])
            ]
            for action, agent in zip(local_actions, self.agents)
        ]

        nx.set_node_attributes(
            graph, {node: global_actions[node] for node in graph.nodes}, "action"
        )

        subgraphs: List[dgl.DGLHeteroGraph] = [
            from_networkx_to_dgl(graph.subgraph(community), has_action=True)
            for community in self.communities
        ]

        manager_decisions = [
            manager(subgraph) for manager, subgraph in zip(self.managers, subgraphs)
        ]
        managed_actions = [
            int(manager_decision[0]) for manager_decision in manager_decisions
        ]
        print("Managed Actions: " + str(managed_actions))
        embedded_graphs = [
            manager_decision[1] for manager_decision in manager_decisions
        ]
        print("Embedded Graphs:")
        for idx, embedded_graph in enumerate(embedded_graphs):
            print(embedded_graph)

        summarized_graph = self.summarize_graph(graph, embedded_graphs, managed_actions)
        best_action, _ = self.head_manager(summarized_graph)
        return best_action

    def summarize_graph(
        self,
        graph: nx.Graph,
        embedded_graphs: List[dgl.DGLHeteroGraph],
        managed_actions: List[int],
    ) -> dgl.DGLHeteroGraph:
        # TODO: implement an Abstract Manager
        # TODO: Manager deals with both Node and Edge Features
        # TODO: Head Manager deals only with Node Features
        # TODO: Internal Edge Features are included into node features through convolution

        node_attribute_dict = {}
        for subgraph, community in zip(embedded_graphs, self.communities):
            for idx, node in enumerate(sorted(list(community))):
                node_attribute_dict[node] = {
                    "embedding": subgraph.nodes[idx].data["embedding"].detach()
                }

        nx.set_node_attributes(
            graph,
            node_attribute_dict,
        )
        summarized_graph: nx.graph = graph
        for community in self.communities:
            community_list = list(community)
            for node in community_list[1:]:
                summarized_graph = nx.contracted_nodes(
                    summarized_graph, community_list[0], node
                )
        for node_data in summarized_graph.nodes.data():
            node_data[1]["embedding"] = th.mean(
                th.stack(
                    [
                        node_data[1]["contraction"][contracted_node]["embedding"]
                        for contracted_node in list(node_data[1]["contraction"].keys())
                    ]
                ),
                dim=-1,
            )
            for idx, community in enumerate(self.communities):
                if node_data[0] in community:
                    node_data[1]["action"] = managed_actions[idx]
            for key in list(node_data[1].keys()):
                if key != "embedding" and key != "action":
                    del node_data[1][key]

        for edge in graph.edges.data():
            if edge[2].get("contraction"):
                edge[2]["embedding"] = th.mean(
                    th.stack(
                        [
                            edge[2][contracted_edge]["embedding"]
                            for contracted_edge in list(edge[2]["contraction"].keys())
                        ]
                    ),
                    dim=-1,
                )
            for key in list(edge[2].keys()):
                if key != "embedding":
                    del edge[2][key]

        print(summarized_graph.edges.data())
        return dgl.from_networkx(
            summarized_graph.to_directed(),
            node_attrs=["action", "embedding"],
            edge_attrs=["embedding"],
        )

    def convert_obs(
        self, observation: BaseObservation
    ) -> Tuple[List[dgl.DGLHeteroGraph], nx.Graph]:
        return factor_observation(observation, self.architecture["radius"])

    @staticmethod
    def get_graph(observation: BaseObservation) -> nx.Graph:
        return observation.as_networkx()
