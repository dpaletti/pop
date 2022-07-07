from typing import Tuple, Union
from random import choice

import dgl
import networkx as nx
from grid2op.Observation import BaseObservation


def from_networkx_to_dgl(graph, device) -> dgl.DGLHeteroGraph:
    return dgl.from_networkx(
        graph.to_directed(),
        node_attrs=list(graph.nodes[choice(list(graph.nodes))].keys())
        if len(graph.nodes) > 0
        else [],
        edge_attrs=list(graph.edges[choice(list(graph.edges))].keys())
        if len(graph.edges) > 0
        else [],
        device=device,
    )


def to_dgl(obs: BaseObservation, device) -> dgl.DGLHeteroGraph:

    # Convert Grid2op graph to a directed (for compatibility reasons) networkx graph
    net = obs.as_networkx()
    net = net.to_directed()  # Typing error from networkx, ignore it

    # Convert from networkx to dgl graph
    return from_networkx_to_dgl(net, device)


def batch_observations(
    observations: Union[Tuple[BaseObservation, ...], Tuple[nx.Graph, ...]], device
) -> dgl.DGLHeteroGraph:

    graphs = []
    for o in observations:
        graph = (
            to_dgl(o, device)
            if isinstance(o, BaseObservation)
            else from_networkx_to_dgl(o, device)
        )
        graphs.append(graph)
    graph_batch = dgl.batch(graphs)
    return graph_batch
