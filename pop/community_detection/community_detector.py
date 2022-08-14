from random import Random, sample
from typing import List, Set, Tuple, Optional, FrozenSet

import networkx as nx
import networkx.linalg as nx_linalg
import numpy as np
from pop.community_detection.louvain import louvain_communities
from pop.community_detection.power_supply_modularity import belong_to_same_community


Community = FrozenSet[int]


class CommunityDetector:
    def __init__(
        self,
        seed: int,
        resolution: float = 1.0,
        threshold: float = 1e-6,
        enable_power_supply_modularity=False,
    ) -> None:
        self.resolution: float = resolution
        self.threshold: float = threshold
        self.seed: int = seed
        self.enable_power_supply_modularity: bool = enable_power_supply_modularity

    @staticmethod
    def community_coherence(graph: nx.Graph, community: Set[int]) -> int:
        adjacency_matrix = nx_linalg.adjacency_matrix(graph)
        return sum([adjacency_matrix[i, j] for i, j in zip(community, community)])

    @staticmethod
    def community_degree(graph: nx.Graph, community: Set[int]) -> int:
        return sum([graph.degree[i] for i in community])

    @staticmethod
    def get_community(node: int, communities: List[Set[int]]) -> Set[int]:
        for community in communities:
            if node in community:
                return community
        raise Exception("Could not find " + str(node) + " in any community")

    def initialize_intermediate_community_structure(
        self, graph_t: nx.Graph, graph_t1: nx.Graph, comm_t: List[Set[int]]
    ) -> Tuple[Set[Set[int]], Set[Tuple[int, int]]]:
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
        singleton_communities: set = set()  # C1
        two_vertices_communities: set = set()  # C2

        added_edges: set = set(graph_t1.edges - graph_t.edges)
        removed_edges: set = set(graph_t.edges - graph_t1.edges)

        added_nodes: set = set(graph_t1.nodes - graph_t.nodes)
        removed_nodes: set = set(graph_t.nodes - graph_t1.nodes)

        for edge in added_edges.union(removed_edges):
            for k in edge:
                if k in removed_nodes:
                    removed_node_community = self.get_community(k, comm_t).remove(k)
                    singleton_communities.add(frozenset(removed_node_community))
                    for old_edge in [
                        e
                        for e in graph_t.edges
                        if ((e[0] == k or e[1] == k) and e[0] != e[1])
                    ]:
                        singleton_communities.add(
                            frozenset(
                                self.get_community(
                                    old_edge[1] if old_edge[1] != k else old_edge[0],
                                    comm_t,
                                )
                            )
                        )
                if k in added_nodes:
                    singleton_communities.add(frozenset({k}))

                    new_edges = [
                        e
                        for e in graph_t1.edges
                        if ((e[0] == k or e[1] == k) and e[0] != e[1])
                    ]
                    if new_edges:
                        # new_edges may be empty and induce an exception
                        # sampling is used to handle the unweighted case
                        two_vertices_communities.add(sample(new_edges, 1)[0])

                    for new_edge in new_edges:
                        singleton_communities.add(
                            frozenset(
                                self.get_community(
                                    new_edge[1] if new_edge[1] != k else new_edge[0],
                                    comm_t,
                                )
                            )
                        )
            if edge[0] not in removed_nodes.union(added_nodes) and edge[
                1
            ] not in removed_nodes.union(added_nodes):
                if edge in removed_edges:
                    if belong_to_same_community(edge[0], edge[1], comm_t):
                        singleton_communities.add(
                            frozenset(self.get_community(edge[0], comm_t))
                        )  # edge[0] and edge[1] belong to same community
                        for extremity in edge:
                            for neighbor in graph_t.neighbors(extremity):
                                singleton_communities.add(
                                    frozenset(self.get_community(neighbor, comm_t))
                                )

                if edge in added_edges:
                    if belong_to_same_community(edge[0], edge[1], comm_t):
                        two_vertices_communities.add(edge)
                        singleton_communities.add(
                            frozenset(self.get_community(edge[0], comm_t))
                        )
                    else:
                        edge_origin_community = self.get_community(edge[0], comm_t)
                        edge_extremity_community = self.get_community(edge[1], comm_t)
                        delta_w = 1  # we deal only with unweighted edges
                        merged_community = set.union(
                            edge_origin_community,
                            edge_extremity_community,
                        )
                        coherence_0 = self.community_coherence(
                            graph_t, edge_origin_community
                        )
                        coherence_1 = self.community_coherence(
                            graph_t, edge_extremity_community
                        )
                        coherence_merged = self.community_coherence(
                            graph_t, merged_community
                        )

                        degree_0 = self.community_degree(graph_t, edge_origin_community)
                        degree_1 = self.community_degree(
                            graph_t, edge_extremity_community
                        )

                        coherence_delta = coherence_0 + coherence_1 - coherence_merged
                        full_degree = degree_0 + degree_1
                        m = len(graph_t.edges)

                        delta_1 = 2 * m - coherence_delta - full_degree
                        delta_2 = m * coherence_delta + degree_0 * degree_1

                        if 2 * delta_w + delta_1 > np.sqrt(
                            delta_1**2 + 4 * delta_2**2
                        ):
                            singleton_communities.add(frozenset(edge_origin_community))
                            singleton_communities.add(
                                frozenset(edge_extremity_community)
                            )
                            two_vertices_communities.add(edge)

        return singleton_communities, two_vertices_communities

    def dynamo(
        self,
        graph_t: nx.Graph,
        graph_t1: Optional[nx.Graph] = None,
        comm_t: Optional[List[Set[int]]] = None,
        alpha: float = 0.5,
        beta: float = 0.5,
    ) -> List[FrozenSet[int]]:
        """
        Two phases:
        - initialize an intermediate community structure
        - repeat the last two steps of Louvain algorithm on the intermediate
          community structure until the modularity gain is negligible
        """
        if not comm_t and not graph_t1:
            return [
                frozenset(community)
                for community in louvain_communities(
                    graph_t,
                    [{i} for i in graph_t.nodes],
                    weight=None,
                    resolution=self.resolution,
                    threshold=self.threshold,
                    seed=Random(self.seed),
                    enable_power_supply_modularity=self.enable_power_supply_modularity,
                    alpha=alpha,
                    beta=beta,
                )
            ]
        if (not comm_t and graph_t1) or (comm_t and not graph_t1):
            raise Exception(
                "comm_t and graph_t1 must be either both None or both not None"
            )

        comm_t = [set(community) for community in comm_t]
        (
            singleton_communities,
            two_vertices_communities,
        ) = self.initialize_intermediate_community_structure(graph_t, graph_t1, comm_t)
        comm_t1: List[Set[int]] = [i.copy() for i in comm_t]

        for singleton_community in singleton_communities:
            comm_t1 = list(
                filter(lambda community: singleton_community != community, comm_t1)
            )
            for node in singleton_community:
                comm_t1.append({node})

        for two_vertices_community in two_vertices_communities:
            comm_t1 = list(
                filter(
                    lambda community: two_vertices_community[0] not in community
                    and two_vertices_community[1] not in community,
                    comm_t1,
                )
            )
            comm_t1.append(set(two_vertices_community))

        if len([node for community in comm_t1 for node in community]) != len(
            graph_t1.nodes
        ) or set([node for community in comm_t1 for node in community]) != set(
            graph_t1.nodes
        ):
            # TODO: remove
            print("A STOPPING POINT")
        return [
            frozenset(community)
            for community in louvain_communities(
                graph_t1,
                comm_t1,
                weight=None,
                resolution=self.resolution,
                threshold=self.threshold,
                seed=Random(self.seed),
                enable_power_supply_modularity=self.enable_power_supply_modularity,
                alpha=alpha,
                beta=beta,
            )
        ]
