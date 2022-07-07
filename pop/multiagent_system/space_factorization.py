import hashlib

import dgl
from grid2op.Action import BaseAction
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation
from typing import List, Tuple, Optional, Dict, Set
import networkx as nx
import numpy as np
import torch as th

from pop.agents.utilities import from_networkx_to_dgl


class HashableAction:
    def __init__(self, action: BaseAction):
        self.action = action
        self.hash = hashlib.md5()

    def __key(self):
        return str(self.action.impact_on_objects()).encode()

    def __hash__(self):
        self.hash.update(self.__key())
        return int(self.hash.hexdigest(), base=16)

    def __eq__(self, other):
        if isinstance(other, HashableAction):
            return self.__key() == other.__key()
        return NotImplemented

    def __str__(self):
        return str(self.action)


def _get_topological_action_owner(
    topological_object: Dict[str, str],
    load_to_node: np.array,
    generator_to_node: np.array,
    line_origin_to_node: np.array,
    line_extremity_to_node: np.array,
):
    if topological_object["object_type"] == "line (origin)":
        return line_origin_to_node[topological_object["object_id"]]
    elif topological_object["object_type"] == "line (extremity)":
        return line_extremity_to_node[topological_object["object_id"]]
    elif topological_object["object_type"] == "load":
        return load_to_node[topological_object["object_id"]]
    elif topological_object["object_type"] == "generator":
        return generator_to_node[topological_object["object_id"]]
    else:
        raise Exception("Unkown object_type: " + str(topological_object["object_type"]))


def _assign_action(
    action: BaseAction,
    load_to_node: np.array,
    generator_to_node: np.array,
    line_origin_to_node: np.array,
    line_extremity_to_node: np.array,
    encoded_action: int,
) -> Optional[Tuple[int, BaseAction, int]]:
    (
        _,
        _,
        topology,
        line,
        redispatching,
        _,
        curtailment,
    ) = action.get_types()
    action_impact = action.impact_on_objects()

    if [topology, line, redispatching, curtailment].count(True) > 1:
        # Composite actions are ignored
        return

    if topology:
        action_impact_on_topology = action_impact["topology"]
        if (
            len(action_impact_on_topology) == 1
            and not action_impact_on_topology["assigned_bus"]
        ):
            # Topological switches which include only 1 action
            return (
                _get_topological_action_owner(
                    action_impact_on_topology["bus_switch"][0],
                    load_to_node,
                    generator_to_node,
                    line_origin_to_node,
                    line_extremity_to_node,
                ),
                action,
                encoded_action,
            )
    if line:
        if (
            action_impact["switch_line"]["count"] == 1
            and not list(action_impact["force_line"]["reconnections"]["powerlines"])
            and not list(action_impact["force_line"]["disconnections"]["powerlines"])
        ):
            # Switches are taken into account
            return (
                line_origin_to_node[
                    list(action_impact["switch_line"]["powerlines"])[0]
                ],
                action,
                encoded_action,
            )
    if redispatching:
        for generator in action_impact["redispatch"]["generators"]:
            return generator_to_node[generator["gen_id"]], action, encoded_action


def factor_action_space(
    env: BaseEnv,
) -> Tuple[List[List[int]], Dict[HashableAction, int]]:

    # Initializing converter and reference observation
    full_converter: IdToAct = IdToAct(env.action_space)
    full_converter.init_converter()
    obs = env.reset()
    graph = obs.as_networkx()

    # Here we retrieve the mappings between objects and buses (aka nodes)
    _, mappings = obs.flow_bus_matrix()

    # Unpacking the mappings for each object type
    (
        load_to_node,
        generator_to_node,
        _,
        line_origin_to_node,
        line_extremity_to_node,
    ) = mappings

    # Factoring Action Lookup Table
    owner, actions, encoded_actions = zip(
        *[
            action_assignment
            for action_assignment in [
                _assign_action(
                    action,
                    load_to_node,
                    generator_to_node,
                    line_origin_to_node,
                    line_extremity_to_node,
                    encoded_action,
                )
                for encoded_action, action in enumerate(full_converter.all_actions[1:])
            ]
            if action_assignment is not None
        ]
    )

    action_spaces = [
        [full_converter.all_actions[0]]
        + [action for owner, action in zip(owner, actions) if owner == node]
        for node in graph.nodes
    ]

    return action_spaces, {
        HashableAction(full_converter.all_actions[encoded_action]): encoded_action
        for encoded_action in encoded_actions
    }


def _factor_observation_helper_ego_graph(graph, node, radius, device):
    return from_networkx_to_dgl(nx.ego_graph(graph, node, radius), device)


def _add_self_loop(zero_edges_graph, feature_schema, device):
    lone_node = list(zero_edges_graph.nodes)[0]
    zero_edges_graph.add_edge(
        lone_node,
        lone_node,
        **{
            feature_name: np.float32(0)
            if feature_value.dtype == th.float
            else np.int32(0)
            if feature_value.dtype == th.int
            else False
            for feature_name, feature_value in feature_schema.items()
        }
    )
    return from_networkx_to_dgl(zero_edges_graph, device)


def factor_observation(
    obs: BaseObservation, device, radius: int = 1
) -> Tuple[List[dgl.DGLHeteroGraph], nx.Graph]:

    graph: nx.Graph = obs.as_networkx()
    sub_graphs: List[dgl.DGLHeteroGraph] = []
    zero_edges_ego_graphs: List[nx.Graph] = []
    positions = []

    for idx, node in enumerate(graph.nodes):
        ego_graph: nx.Graph = nx.ego_graph(graph, node, radius)

        if ego_graph.number_of_edges() == 0:
            # Zero edges ego-graphs must have at least a self loop
            positions.append(idx)
            print(
                "WARNING: found zero edges ego graph, adding self-loop and zeroed-out features for consistency"
            )
            print(
                "WARNING: features are assumed to have 32-bit precision and be either int, float or bool"
            )
            zero_edges_ego_graphs.append(ego_graph)
            continue

        subgraph = from_networkx_to_dgl(ego_graph, device)
        sub_graphs.append(subgraph)

    if not sub_graphs:
        # At least 1 ego graph must be non-degenerate
        for ego_graph in zero_edges_ego_graphs:
            print(ego_graph)
        raise Exception("Found only ego graphs with zero features")

    if zero_edges_ego_graphs:
        # Fixing zero edges ego graphs
        feature_schema = sub_graphs[0].edata
        for ego_graph, position in zip(zero_edges_ego_graphs, positions):
            sub_graphs.insert(
                position, _add_self_loop(ego_graph, feature_schema, device)
            )

    return sub_graphs, graph


def split_graph_into_communities(
    graph: nx.Graph, communities: List[Tuple[int, ...]], device
) -> Dict[Tuple[int, ...], dgl.DGLHeteroGraph]:

    nx_sub_graphs: List[nx.Graph] = []
    zero_edges_nx_sub_graphs: List[nx.Graph] = []
    positions = []

    # Split into one subgraph for each community
    for idx, community in enumerate(communities):
        sub_g = graph.subgraph(community)

        # If subgraph does not have any edge
        if sub_g.number_of_edges == 0:
            positions.append(idx)
            zero_edges_nx_sub_graphs.append(sub_g)
        else:
            nx_sub_graphs.append(sub_g)

    # Convert correct sub_graphs to dgl
    sub_graphs: List[dgl.DGLHeteroGraph] = [
        from_networkx_to_dgl(subgraph, device) for subgraph in nx_sub_graphs
    ]

    # Add self loop to 0 edges subgraphs
    feature_schema = sub_graphs[0].edata
    for zero_edge_sub_graph, position in zip(zero_edges_nx_sub_graphs, positions):
        sub_graphs.insert(
            position,
            _add_self_loop(zero_edge_sub_graph, feature_schema, device),
        )

    return {
        community: sub_graph for community, sub_graph in zip(communities, sub_graphs)
    }
