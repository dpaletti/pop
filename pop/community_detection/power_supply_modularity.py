import networkx as nx
import numpy as np
from typing import Tuple, List, Set


def belong_to_same_community(node1: int, node2: int, communities: List[Set[int]]):
    for community in communities:
        if node1 in community and node2 in community:
            return True
    return False


def compute_reactance(graph: nx.Graph, edge: Tuple[int, int]):
    reactive_power: float = graph.edges[edge]["q_or"] + graph.edges[edge]["q_ex"]
    current_flow: float = graph.edges[edge]["a_or"] + graph.edges[edge]["a_ex"]
    return reactive_power / current_flow**2


def compute_admittance_matrix(graph: nx.Graph):
    return np.array(
        [
            [
                0 if (i, j) not in graph.edges else 1 / compute_reactance(graph, (i, j))
                for i in graph.nodes
            ]
            for j in graph.nodes
        ]
    )


def compute_line_admittance_matrix(graph: nx.Graph):
    return np.array(
        [
            [
                1 / compute_reactance(graph, edge)
                if node == edge[0]
                else -1 / compute_reactance(graph, edge)
                if node == edge[1]
                else 0
                for node in graph.nodes
            ]
            for edge in graph.edges
        ]
    )


def compute_nodal_admittance_matrix(graph: nx.Graph, slack_node: int):
    # We assume that reactance is far larger than resistance
    # Thus we ignore resistance when computing admittance

    return np.array(
        [
            [
                sum(
                    [
                        (1 / compute_reactance(graph, (i, neighbor)))
                        if neighbor > i
                        else (1 / compute_reactance(graph, (neighbor, i)))
                        for neighbor in list(graph.neighbors(i))
                    ]
                )
                if i == j
                else 0
                if (i, j) not in graph.edges or i == slack_node or j == slack_node
                else (-1 / compute_reactance(graph, (i, j)))
                for i in graph.nodes
            ]
            for j in graph.nodes
        ]
    )


def compute_power_transfer_distribution_factor(graph: nx.Graph):
    if not nx.is_connected(graph):
        nodal_admittances = [
            np.linalg.pinv(compute_nodal_admittance_matrix(graph, node))
            for node in graph.nodes
        ]
    else:
        nodal_admittances = [
            np.linalg.inv(compute_nodal_admittance_matrix(graph, node))
            for node in graph.nodes
        ]
    line_admittance_matrix = compute_line_admittance_matrix(graph)
    return np.stack(
        [
            np.matmul(line_admittance_matrix, nodal_admittance_matrix)
            for nodal_admittance_matrix in nodal_admittances
        ],
        axis=2,
    )


def compute_power_transmission_capacity(graph: nx.Graph):
    ptdf = compute_power_transfer_distribution_factor(graph)
    powerflow_limit = np.array(
        [
            (graph.edges[edge]["a_or"] + graph.edges[edge]["a_ex"])
            / graph.edges[edge]["rho"]
            for edge in graph.edges
        ]
    )
    return np.array(
        [
            [min(powerflow_limit / np.linalg.norm(ptdf[:, i, j])) for i in graph.nodes]
            for j in graph.nodes
        ]
    )


def compute_electrical_coupling_strength(
    graph: nx.Graph, alpha: float = 0.5, beta: float = 0.5
):
    power_transmission_capacity = compute_power_transmission_capacity(graph)
    normalized_power_transmission_capacity = power_transmission_capacity / np.mean(
        power_transmission_capacity
    )

    admittance_matrix = compute_admittance_matrix(graph)
    normalized_admittance_matrix = admittance_matrix / np.mean(admittance_matrix)

    return np.sqrt(
        np.power(alpha * normalized_power_transmission_capacity, 2)
        + np.power(beta * normalized_admittance_matrix, 2)
    )


def power_supply_modularity(
    graph: nx.Graph,
    community: List[Set[int]],
    alpha: float = 0.5,
    beta: float = 0.5,
):
    ecs = compute_electrical_coupling_strength(graph, alpha, beta)
    total_ecs = 2 * np.sum(ecs)  # we will only use this formulation
    return np.sum(
        [
            [
                (
                    ecs[i, j] / total_ecs
                    - (ecs[i, :] / total_ecs) * (ecs[j, :] / total_ecs)
                )
                * int(belong_to_same_community(i, j, community))
                for i in graph.nodes
            ]
            for j in graph.nodes
        ]
    )
