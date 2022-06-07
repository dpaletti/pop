from typing import Tuple, Union
from random import choice, sample

import dgl
import networkx as nx
from grid2op.Observation import BaseObservation
import numpy as np
import torch as th


def from_networkx_to_dgl(graph, device) -> dgl.DGLHeteroGraph:
    try:
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
    except Exception as e:
        if type(graph) is dgl.DGLHeteroGraph:
            return graph
        else:
            print("Hit Exception in from_networkx_to_dgl")
            print(graph)
            print(graph.nodes)
            print(graph.edges)
            raise e


def to_dgl(obs: BaseObservation, device) -> dgl.DGLHeteroGraph:

    # Convert Grid2op graph to a directed (for compatibility reasons) networkx graph
    net = obs.as_networkx()
    net = net.to_directed()  # Typing error from networkx, ignore it

    # Convert from networkx to dgl graph
    return from_networkx_to_dgl(net, device)


def batch_observations(
    observations: Union[Tuple[BaseObservation], Tuple[nx.Graph]], device
) -> dgl.DGLHeteroGraph:

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


def add_self_loop(zero_edges_graph, feature_schema, device):
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
