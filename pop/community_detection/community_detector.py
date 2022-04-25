from random import Random
from typing import List, Set, Tuple, Optional

import networkx as nx
import networkx.linalg as nx_linalg
import numpy as np
from community_detection.louvain import louvain_communities
from power_supply_modularity import belong_to_same_community


class CommunityDetector:
    def __init__(
        self, seed: int, resolution: float = 1.0, threshold: float = 1e-6
    ) -> None:
        self.resolution: float = resolution
        self.threshold: float = threshold
        self.seed: int = seed

    @staticmethod
    def community_coherence(graph: nx.Graph, community: Set[int]) -> int:
        adjacency_matrix = nx_linalg.adjacency_matrix(graph)
        return sum([adjacency_matrix[i, j] for i, j in zip(community, community)])

    @staticmethod
    def community_degree(graph: nx.Graph, community: Set[int]) -> int:
        return sum([graph.degree[i] for i in community])

    @staticmethod
    def get_community(node_1: int, communities: List[Set[int]]) -> int:
        for community_id, community in enumerate(communities):
            if node_1 in community:
                return community_id
        raise Exception(
            "Node: "
            + str(node_1)
            + " does not belong to any community\n "
            + str(communities)
        )

    def initialize_intermediate_community_structure(
        self, graph_t: nx.Graph, graph_t1: nx.Graph, comm_t: List[Set[int]]
    ) -> Tuple[Set[int], Set[Tuple[int, int]]]:
        """
        Takes C_t, community structure at time t, and modifies it by selecting singletons and two-vertices
        communities.
        Node structure does not change in our case thus we skip it, we only consider edge addition/deletion
        :param graph_t: graph at previous timestep
        :param graph_t1: current graph
        :param comm_t: community structure at previous timestep
        :return
        - C_1: a set of communities to be separated into singleton communities
        - C_2: a set of two-vertices communities to be created
        """
        singleton_communities = set()
        two_verticies_communities = set()
        added_edges = nx.difference(graph_t1, graph_t).edges
        removed_edges = nx.difference(graph_t, graph_t1).edges
        for edge in removed_edges:
            if belong_to_same_community(edge[0], edge[1], comm_t):
                singleton_communities.add(
                    self.get_community(edge[0], comm_t)
                )  # edge[0] and edge[1] belong to same community
                for extremity in edge:
                    for neighbor in graph_t.neighbors(extremity):
                        singleton_communities.add(self.get_community(neighbor, comm_t))

        for edge in added_edges:
            if belong_to_same_community(edge[0], edge[1], comm_t):
                two_verticies_communities.add(edge)
                singleton_communities.add(self.get_community(edge[0], comm_t))
            else:
                delta_w = 1  # we deal only with unweighted edges
                merged_community = set.union(
                    comm_t[self.get_community(edge[0], comm_t)],
                    comm_t[self.get_community(edge[1], comm_t)],
                )
                coherence_0 = self.community_coherence(
                    graph_t, comm_t[self.get_community(edge[0], comm_t)]
                )
                coherence_1 = self.community_coherence(
                    graph_t, comm_t[self.get_community(edge[1], comm_t)]
                )
                coherence_merged = self.community_coherence(graph_t, merged_community)

                degree_0 = self.community_degree(
                    graph_t, comm_t[self.get_community(edge[0], comm_t)]
                )
                degree_1 = self.community_degree(
                    graph_t, comm_t[self.get_community(edge[1], comm_t)]
                )

                coherence_delta = coherence_0 + coherence_1 - coherence_merged
                full_degree = degree_0 + degree_1
                m = len(graph_t.edges)

                delta_1 = 2 * m - coherence_delta - full_degree
                delta_2 = m * coherence_delta + degree_0 * degree_1

                if 2 * delta_w + delta_1 > np.sqrt(delta_1**2 + 4 * delta_2**2):
                    singleton_communities.add(self.get_community(edge[0], comm_t))
                    singleton_communities.add(self.get_community(edge[1], comm_t))
                    two_verticies_communities.add(edge)

        return singleton_communities, two_verticies_communities

    def dynamo(
        self,
        graph_t: nx.Graph,
        graph_t1: Optional[nx.Graph] = None,
        comm_t: Optional[List[Set[int]]] = None,
        enable_power_supply_modularity=False,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> List[Set[int]]:
        """
        Two phases:
        - initialize an intermediate community structure
        - repeat the last two steps of Louvain algorithm on the intermediate
          community structure until the modularity gain is negligible

        :param graph_t: graph structure at previous time step
        :param comm_t: previous community structure
        :param graph_t1: new graph
        :return: community structure
        """
        if not comm_t and not graph_t1:
            return louvain_communities(
                graph_t,
                [{i} for i in graph_t.nodes],
                weight=None,
                resolution=self.resolution,
                threshold=self.threshold,
                seed=Random(self.seed),
                enable_power_supply_modularity=enable_power_supply_modularity,
                alpha=alpha,
                beta=beta,
            )
        if (not comm_t and graph_t1) or (comm_t and not graph_t1):
            raise Exception(
                "comm_t and graph_t1 must be either both None or both not None"
            )

        (
            singleton_communities,
            two_vertices_communities,
        ) = self.initialize_intermediate_community_structure(graph_t, graph_t1, comm_t)
        comm_t1: List[Optional[Set[int]]] = [i.copy() for i in comm_t]

        for singleton_community in singleton_communities:
            comm_t1[singleton_community] = None
            for node in comm_t[singleton_community]:
                if (
                    {node} not in comm_t1
                    and node not in [t[0] for t in two_vertices_communities]
                    and node not in [t[1] for t in two_vertices_communities]
                ):
                    comm_t1.append({node})

        for two_vertices_community in two_vertices_communities:
            comm_t1[self.get_community(two_vertices_community[0], comm_t)] = None
            comm_t1[self.get_community(two_vertices_community[1], comm_t)] = None
            comm_t1.append(set(two_vertices_community))

        comm_t1 = [i for i in comm_t1 if i]
        return louvain_communities(
            graph_t1,
            comm_t1,
            weight=None,
            resolution=self.resolution,
            threshold=self.threshold,
            seed=Random(self.seed),
            enable_power_supply_modularity=enable_power_supply_modularity,
            alpha=alpha,
            beta=beta,
        )
