"""
This type stub file was generated by pyright.
"""

import networkx as nx
from typing import List, Set, Tuple

def belong_to_same_community(node1: int, node2: int, communities: List[Set[int]]): # -> bool:
    ...

def compute_reactance(graph: nx.Graph, edge: Tuple[int, int]): # -> float:
    ...

def compute_admittance_matrix(graph: nx.Graph): # -> ndarray[Unknown, Unknown]:
    ...

def compute_line_admittance_matrix(graph: nx.Graph): # -> ndarray[Unknown, Unknown]:
    ...

def compute_nodal_admittance_matrix(graph: nx.Graph, slack_node: int): # -> ndarray[Unknown, Unknown]:
    ...

def compute_power_transfer_distribution_factor(graph: nx.Graph): # -> ndarray[Unknown, Unknown]:
    ...

def compute_power_transmission_capacity(graph: nx.Graph): # -> ndarray[Unknown, Unknown]:
    ...

def compute_electrical_coupling_strength(graph: nx.Graph, alpha: float = ..., beta: float = ...): # -> Any:
    ...

def power_supply_modularity(graph: nx.Graph, community: List[Set[int]], alpha: float = ..., beta: float = ...): # -> Any:
    ...

