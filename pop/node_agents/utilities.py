from typing import Tuple

import dgl
import networkx as nx
from grid2op.Observation import BaseObservation


def from_networkx_to_dgl(graph: nx.graph, has_action=False):
    return dgl.from_networkx(
        graph.to_directed(),
        node_attrs=["p", "q", "v", "cooldown"]
        if not has_action
        else ["p", "q", "v", "cooldown", "action"],
        edge_attrs=[
            "rho",
            "cooldown",
            "status",
            "thermal_limit",
            "timestep_overflow",
            "p_or",
            "p_ex",
            "q_or",
            "q_ex",
            "a_or",
            "a_ex",
        ],
    )


def to_dgl(obs: BaseObservation) -> dgl.DGLHeteroGraph:
    """
    convert a :class:BaseObservation to a :class:`dgl.DGLHeteroGraph`.

    Parameters
    ----------
    obs: :class:`BaseObservation`
        BaseObservation taken from a grid2Op environment

    Return
    ------
    dgl_obs: :class:`dgl.DGLHeteroGraph`
        graph compatible with the Deep Graph Library
    """

    # Convert Grid2op graph to a directed (for compatibility reasons) networkx graph
    net = obs.as_networkx()
    net = net.to_directed()  # Typing error from networkx, ignore it

    # Convert from networkx to dgl graph
    dgl_net = dgl.from_networkx(
        net,
        node_attrs=["p", "q", "v", "cooldown"],
        edge_attrs=[
            "rho",
            "cooldown",
            "status",
            "thermal_limit",
            "timestep_overflow",
            "p_or",
            "p_ex",
            "q_or",
            "q_ex",
            "a_or",
            "a_ex",
        ],
    )
    return dgl_net


def batch_observations(observations: Tuple[BaseObservation]) -> dgl.DGLHeteroGraph:
    """
    Convert a list (or tuple) of observations to a Deep Graph Library graph batch.
    A graph batch is represented as a normal graph with nodes and edges added (together with features).

    Parameters
    ----------
    observations: ``Tuple[BaseObservation]``
        tuple of BaseObservation usually stored inside of a :class:`Transition`

    Return
    ------
    graph_batch: :class:`dgl.DGLHeteroGraph`
        a batch of graphs represented as a single augmented graph for Deep Graph Library compatibility
    """
    graphs = []
    for o in observations:
        graph = to_dgl(o)
        graphs.append(graph)
    graph_batch = dgl.batch(graphs)
    return graph_batch
