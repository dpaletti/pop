import hashlib

import dgl
import torch as th
from grid2op.Action import BaseAction
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation
from typing import List, Tuple, Optional
import networkx as nx
import numpy as np

from pop.node_agents.utilities import from_networkx_to_dgl, add_self_loop


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


def _add_topological_action_to_node(bus, load, prod, ind_lor, ind_lex):
    if bus["object_type"] == "line (origin)":
        return ind_lor[bus["object_id"]]
    elif bus["object_type"] == "line (extremity)":
        return ind_lex[bus["object_id"]]
    elif bus["object_type"] == "load":
        return load[bus["object_id"]]
    elif bus["object_type"] == "generator":
        return prod[bus["object_id"]]
    else:
        raise Exception("Unkown object_type: " + str(bus["object_type"]))


def assign_action(
    action, load, prod, ind_lor, ind_lex, idx
) -> Tuple[Optional[int], BaseAction, int]:
    # Injection, Voltage and Storage are ignored
    # could not find it in the L2RPN action space
    # assuming they are not available in general environments
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
    action_owners = set()
    if topology:
        action_impact_on_topology = action_impact["topology"]
        if action_impact_on_topology["assigned_bus"]:
            return None, action, idx

        if len(action_impact_on_topology) == 1:
            action_owners.add(
                _add_topological_action_to_node(
                    action_impact_on_topology["bus_switch"][0],
                    load,
                    prod,
                    ind_lor,
                    ind_lex,
                )
            )
        else:
            return None, action, idx
    if line:
        # We assign powerlines to the origin node by convention
        if list(action_impact["force_line"]["reconnections"]["powerlines"]):
            return None, action, idx
        if list(action_impact["force_line"]["disconnections"]["powerlines"]):
            return None, action, idx
        if action_impact["switch_line"]["count"] == 1:
            action_owners.add(
                ind_lor[list(action_impact["switch_line"]["powerlines"])[0]]
            )
        else:
            return None, action, idx
    if redispatching:
        for generator in action_impact["redispatch"]["generators"]:
            action_owners.add(prod[generator["gen_id"]])
    try:
        return action_owners.pop(), action, idx
    except:
        return None, action, idx


def factor_action_space(env: BaseEnv):
    full_converter: IdToAct = IdToAct(env.action_space)
    full_converter.init_converter()
    obs = env.reset()
    graph = obs.as_networkx()
    mat, (load, prod, _, ind_lor, ind_lex) = obs.flow_bus_matrix()

    # ("Injection, Voltage and Storage actions are ignored.")
    # ("Could not find any in the L2RPN action space.")
    # ("If using a different environment please check availability of such actions")

    # Factoring Action Lookup Table
    owner, actions, idx = zip(
        *[
            assign_action(action, load, prod, ind_lor, ind_lex, idx)
            for idx, action in enumerate(full_converter.all_actions[1:])
        ]
    )

    action_spaces = []
    for bus in graph.nodes:
        action_spaces.append(
            [full_converter.all_actions[0]]
            + [action for owner, action in zip(owner, actions) if owner == bus]
        )

    print("Building Action Lookup Table")

    return action_spaces, {
        HashableAction(action): idx
        for idx, action in enumerate(full_converter.all_actions)
    }


def __factor_observation_helper_ego_graph(graph, node, radius, device):
    return from_networkx_to_dgl(nx.ego_graph(graph, node, radius), device)


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
        for ego_graph in zero_edges_ego_graphs:
            print(ego_graph)
        raise Exception("Found only egographs with zero features")
    if zero_edges_ego_graphs:
        feature_schema = sub_graphs[0].edata
        for ego_graph, position in zip(zero_edges_ego_graphs, positions):
            sub_graphs.insert(
                position, add_self_loop(ego_graph, feature_schema, device)
            )
    return sub_graphs, graph
