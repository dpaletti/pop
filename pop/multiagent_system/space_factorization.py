import hashlib

import dgl
from grid2op.Action import BaseAction
from grid2op.Converter import IdToAct
from grid2op.Observation import BaseObservation
from typing import List, Tuple, Optional, Dict
import networkx as nx
import numpy as np

from agents.base_gcn_agent import BaseGCNAgent
from community_detection.community_detector import Community

Substation = int
EncodedAction = int


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
        raise Exception(
            "Unknown object_type: " + str(topological_object["object_type"])
        )


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
    observation: BaseObservation,
    observation_graph: nx.Graph,
    full_converter: IdToAct,
    n_substations: int,
) -> Tuple[Dict[int, List[int]], Dict[HashableAction, int]]:

    # Here we retrieve the mappings between objects and buses (aka nodes)
    # actual flow_bus_matrix is ignored, useless in this context
    _, mappings = observation.flow_bus_matrix()

    # Unpacking the mappings for each object type
    (
        load_to_node,
        generator_to_node,
        _,  # storage_to_node ignored
        line_origin_to_node,
        line_extremity_to_node,
    ) = mappings

    print(
        "WARNING: Storage objects are ignored, check if they are present in the environment"
    )

    # Factoring Action Lookup Table
    owners, actions, encoded_actions = zip(
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

    action_space_dict = {
        substation: [(full_converter.all_actions[0], 0)]
        for substation in range(n_substations)
    }
    for owner, action, encoded_action in zip(owners, actions, encoded_actions):
        action_space_dict[owner].append((action, encoded_action))

    sub_id_to_action_space = {
        substation: [action[0] for action in action_space_dict[substation]]
        for substation in range(n_substations)
    }

    lookup_table = {}
    for owner, action_mapping_list in action_space_dict.items():
        for action, encoded_action in action_mapping_list:
            lookup_table[HashableAction(action)] = encoded_action

    return sub_id_to_action_space, lookup_table


def factor_observation(
    obs_graph: nx.Graph, device: str, radius: int = 1
) -> Dict[Substation, Optional[dgl.DGLHeteroGraph]]:

    return {
        sub_id: BaseGCNAgent.from_networkx_to_dgl(ego_graph, device)
        if ego_graph.number_of_edges() > 0
        else None
        for sub_id, ego_graph in {
            node_data["sub_id"]: nx.ego_graph(obs_graph, node_id, radius)
            for node_id, node_data in obs_graph.nodes.data()
        }.items()
    }


def split_graph_into_communities(
    graph: nx.Graph, communities: List[Community], device: str
) -> Dict[Community, dgl.DGLHeteroGraph]:

    return {
        community: BaseGCNAgent.from_networkx_to_dgl(graph.subgraph(community), device)
        for community in communities
    }
