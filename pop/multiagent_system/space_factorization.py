import hashlib
import dgl
from grid2op.Action import BaseAction
from grid2op.Converter import IdToAct
from grid2op.Environment import BaseEnv
from grid2op.Observation import BaseObservation
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import List, Tuple, Optional
import networkx as nx

# TODO: The (head) manager should first try to combine the actions
# TODO: Then it should choose which one is better
from node_agents.utilities import from_networkx_to_dgl


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


def factor_action_space(env: BaseEnv, n_jobs: int = 6):
    full_converter: IdToAct = IdToAct(env.action_space)
    full_converter.init_converter()
    obs = env.reset()
    graph = obs.as_networkx()
    mat, (load, prod, _, ind_lor, ind_lex) = obs.flow_bus_matrix()

    print("\nFactoring action space")
    print("Injection, Voltage and Storage actions are ignored.")
    print("Could not find any in the L2RPN action space.")
    print("If using a different environment please check availability of such actions")

    owner, actions, idx = zip(
        *Parallel(n_jobs=n_jobs)(
            delayed(assign_action)(action, load, prod, ind_lor, ind_lex, idx)
            for idx, action in tqdm(enumerate(full_converter.all_actions[1:]))
        )
    )

    action_spaces = []
    for bus in graph.nodes:
        action_spaces.append(
            [full_converter.all_actions[0]]
            + [action for owner, action in zip(owner, actions) if owner == bus]
        )
    print("Building action lookup table")

    return action_spaces, {
        HashableAction(action): idx
        for idx, action in tqdm(enumerate(full_converter.all_actions))
    }


def factor_observation(
    obs: BaseObservation, radius: int = 1
) -> Tuple[List[dgl.DGLHeteroGraph], nx.Graph]:
    graph: nx.Graph = obs.as_networkx()
    sub_graphs: List[dgl.DGLHeteroGraph] = []
    for node in graph.nodes:
        sub_graphs.append(from_networkx_to_dgl(nx.ego_graph(graph, node, radius)))
    return sub_graphs, graph
