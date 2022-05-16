from typing import Tuple, Union
from random import choice, sample

import dgl
import networkx as nx
from grid2op.Observation import BaseObservation


def from_networkx_to_dgl(graph, device) -> dgl.DGLHeteroGraph:
    try:
        return dgl.from_networkx(
            graph.to_directed(),
            node_attrs=list(graph.nodes[choice(list(graph.nodes))].keys()),
            edge_attrs=list(graph.edges[choice(list(graph.edges))].keys()),
            device=device,
        )
    except Exception as e:
        if type(graph) is dgl.DGLHeteroGraph:
            return graph
        else:
            raise (e)


def to_dgl(obs: BaseObservation, device) -> dgl.DGLHeteroGraph:
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
    return from_networkx_to_dgl(net, device)


def batch_observations(
    observations: Union[Tuple[BaseObservation], Tuple[nx.Graph]], device
) -> dgl.DGLHeteroGraph:
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
        graph = (
            to_dgl(o, device)
            if type(o) is BaseObservation
            else from_networkx_to_dgl(o, device)
        )
        graphs.append(graph)
    graph_batch = dgl.batch(graphs)
    return graph_batch
