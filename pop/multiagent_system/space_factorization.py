import hashlib

import dgl
from grid2op.Action import BaseAction
from grid2op.Converter import IdToAct
from grid2op.Observation import ObservationSpace
from typing import List, Tuple, Optional, Dict
import networkx as nx
import numpy as np

from pop.agents.base_gcn_agent import BaseGCNAgent
from pop.community_detection.community_detector import Community
from grid2op.Exceptions.IllegalActionExceptions import IllegalAction
from tqdm import tqdm

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


def _assign_action_gen_only(
    action: BaseAction,
    generator_to_node: np.array,
    encoded_action: int,
) -> Optional[Tuple[int, int]]:

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
        return

    if redispatching:
        for generator in action_impact["redispatch"]["generators"]:
            return generator_to_node[generator["gen_id"]], encoded_action


def _assign_action(
    action: BaseAction,
    load_to_node: np.array,
    generator_to_node: np.array,
    line_origin_to_node: np.array,
    line_extremity_to_node: np.array,
    encoded_action: int,
    composite_actions: bool = False,
    generator_storage_only: bool = False,
) -> Optional[Tuple[int, int]]:
    (
        _,
        _,
        topology,
        line,
        redispatching,
        storage,
        curtailment,
    ) = action.get_types()
    action_impact = action.impact_on_objects()

    if not composite_actions:
        if [topology, line, redispatching, curtailment].count(True) > 1:
            return

    if redispatching:
        for generator in action_impact["redispatch"]["generators"]:
            return generator_to_node[generator["gen_id"]], encoded_action

    if storage:
        raise Exception("Storage not supported: " + str(action_impact))
    if curtailment:
        raise Exception("Curtailment not supported: " + str(action_impact))

    if not generator_storage_only:
        if topology:
            action_impact_on_topology = action_impact["topology"]
            if (
                len(action_impact_on_topology) == 1 or composite_actions
            ) and not action_impact_on_topology["assigned_bus"]:
                # Topological switches which include only 1 action
                return (
                    _get_topological_action_owner(
                        action_impact_on_topology["bus_switch"][0],
                        load_to_node,
                        generator_to_node,
                        line_origin_to_node,
                        line_extremity_to_node,
                    ),
                    encoded_action,
                )
        if line:
            if (
                (action_impact["switch_line"]["count"] == 1 or composite_actions)
                and not list(action_impact["force_line"]["reconnections"]["powerlines"])
                and not list(
                    action_impact["force_line"]["disconnections"]["powerlines"]
                )
            ):
                # Switches are taken into account
                return (
                    line_origin_to_node[
                        list(action_impact["switch_line"]["powerlines"])[0]
                    ],
                    encoded_action,
                )


def generate_redispatching_action_space(env, actions_per_generator: int = 10):
    print("Generating redispatching action space")
    print("WARNING: ignoring actions on storage")
    sub_to_action_space = {sub: [] for sub in range(env.n_sub)}
    all_actions = []
    lookup_table = {}
    converter = IdToAct(env.action_space)

    curtailment_values = np.linspace(0.1, 0.9, actions_per_generator)
    curtailment_actions_available = True

    print("Available curtailment values\n" + str(curtailment_values))
    for gen, sub in enumerate(env.gen_to_subid):
        if env.gen_redispatchable[gen]:
            redispatching_values = np.linspace(
                -env.gen_max_ramp_down[gen],
                env.gen_max_ramp_up[gen],
                actions_per_generator + 2,
            )[1:-1]
            print(
                "Redispatching values for generator "
                + str(gen)
                + "\n"
                + str(redispatching_values)
            )
            for redispatching_value in redispatching_values:
                action = env.action_space()
                action.redispatch = [(gen, redispatching_value)]
                if (
                    not action.is_ambiguous()[0]
                    and action.impact_on_objects()["has_impact"]
                ):
                    sub_to_action_space[sub].append(action)
                    all_actions.append(action)

        if env.gen_renewable[gen] and curtailment_actions_available:
            try:
                for curtailment_value in curtailment_values:
                    action = env.action_space()
                    action.curtail = [(gen, curtailment_value)]
                    if (
                        not action.is_ambiguous()[0]
                        and action.impact_on_objects()["has_impact"]
                    ):
                        sub_to_action_space[sub].append(action)
                        all_actions.append(action)
            except IllegalAction:
                print("Curtailment Actions are disabled in this environment")
                print("Ignoring renewable generators")
                curtailment_actions_available = False

    for _, action_space in sub_to_action_space.items():
        if not action_space:
            action_space.append(env.action_space({}))

    converter.init_converter(all_actions=all_actions)
    converter_actions = list(converter.all_actions)
    for action in converter_actions:
        lookup_table[HashableAction(action)] = converter_actions.index(action)

    return sub_to_action_space, lookup_table


def factor_action_space(
    observation_space: ObservationSpace,
    full_converter: IdToAct,
    n_substations: int,
    composite_actions: bool = False,
    generator_storage_only: bool = False,
    remove_no_action: bool = False,
) -> Tuple[Dict[int, List[int]], Dict[HashableAction, int]]:
    # TODO: take storage into account, take observation_space.storage_to_subid

    # Mappings
    load_to_node = observation_space.load_to_subid
    generator_to_node = observation_space.gen_to_subid
    line_origin_to_node = observation_space.line_or_to_subid
    line_extremity_to_node = observation_space.line_ex_to_subid

    print(
        "WARNING: Storage objects are ignored, check if they are present in the environment"
    )
    print("Factoring Action Space")
    if not composite_actions and generator_storage_only:
        # Factoring Action Lookup Table
        owners, encoded_actions = zip(
            *[
                action_assignment
                for action_assignment in [
                    _assign_action_gen_only(
                        action,
                        generator_to_node,
                        encoded_action,
                    )
                    for encoded_action, action in enumerate(
                        tqdm(full_converter.all_actions[1:])
                    )
                ]
                if action_assignment is not None
            ]
        )
    else:
        # Factoring Action Lookup Table
        owners, encoded_actions = zip(
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
                        composite_actions=composite_actions,
                        generator_storage_only=generator_storage_only,
                    )
                    for encoded_action, action in enumerate(
                        tqdm(full_converter.all_actions[1:])
                    )
                ]
                if action_assignment is not None
            ]
        )

    action_space_dict = {
        substation: [(full_converter.all_actions[0], 0)] if not remove_no_action else []
        for substation in range(n_substations)
    }

    for owner, encoded_action in zip(owners, encoded_actions):
        action_space_dict[owner].append(
            (full_converter.all_actions[encoded_action], encoded_action)
        )

    for _, action_list in action_space_dict.items():
        if not action_list:
            action_list.append((full_converter.all_actions[0], 0))

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
    obs_graph: nx.Graph,
    node_features: List[str],
    edge_features: List[str],
    device: str,
    radius: int = 1,
) -> Dict[Substation, dgl.DGLHeteroGraph]:
    if radius < 1:
        dgl_graph = BaseGCNAgent.from_networkx_to_dgl(
            graph=obs_graph,
            node_features=node_features,
            edge_features=edge_features,
            device=device,
        )
        return {
            sub_id: dgl_graph
            for sub_id in [
                node_data["sub_id"] for _, node_data in obs_graph.nodes.data()
            ]
        }
    ego_graphs_dict = {}
    for node_id, node_data in obs_graph.nodes.data():
        if node_data["sub_id"] not in ego_graphs_dict:
            ego_graphs_dict[node_data["sub_id"]] = nx.ego_graph(
                obs_graph, node_id, radius
            )
        else:
            # Handles the case in which two buses (= two nodes)
            # belong to the same substation
            ego_graphs_dict[node_data["sub_id"]] = nx.compose(
                ego_graphs_dict[node_data["sub_id"]],
                nx.ego_graph(obs_graph, node_id, radius),
            )

    return {
        sub_id: BaseGCNAgent.from_networkx_to_dgl(
            graph=ego_graph,
            node_features=node_features,
            edge_features=edge_features,
            device=device,
        )
        for sub_id, ego_graph in ego_graphs_dict.items()
    }


def split_graph_into_communities(
    graph: nx.Graph,
    communities: List[Community],
    node_features: List[str],
    edge_features: List[str],
    device: str,
) -> Dict[Community, dgl.DGLHeteroGraph]:

    return {
        community: BaseGCNAgent.from_networkx_to_dgl(
            graph=graph.subgraph(community),
            node_features=node_features,
            edge_features=edge_features,
            device=device,
        )
        for community in communities
    }
