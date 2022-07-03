from typing import Optional, Any, Dict, List, Union, Tuple
import dgl

import torch as th
from pathlib import Path
import re

# This imports must be aliased this way for network instantiation
# Do not remove such aliases before changing the instantiation function
import torch.nn as nn
import pop.networks.custom_layers as cl
import dgl.nn.pytorch as dgl_nn

from dgl import DGLHeteroGraph
from torch import Tensor

from networks.network_architecture_parsing import (
    get_network,
)
from pop.configs.network_architecture import NetworkArchitecture, NetworkLayer
from dataclasses import asdict

from utilities import get_log_file


class GCN(nn.Module):
    def __init__(
        self,
        node_features: int,
        architecture: NetworkArchitecture,
        name: str,
        log_dir: Optional[str] = None,
        edge_features: Optional[int] = None,
    ) -> None:
        super(GCN, self).__init__()

        self.name: str = name

        # Fixed Features
        self.node_features: int = node_features
        self.edge_features: Optional[int] = edge_features

        # Model instantiation
        self.model: nn.Sequential = get_network(
            self, architecture, is_graph_network=True
        )
        self.architecture: NetworkArchitecture = architecture

        # Logging
        self.log_file: Optional[str] = get_log_file(log_dir, name)

    def forward(self, g: DGLHeteroGraph) -> Tensor:
        g = self._add_self_loop_to_batched_graph(g)
        node_embeddings: Tensor

        if self.edge_features is not None:
            # -> (nodes*batch_size, heads, out_node_features)
            node_embeddings = self.model(
                g, self._to_tensor(dict(g.ndata)), self._to_tensor(dict(g.edata))
            )

        else:
            # -> (nodes*batch_size, heads, out_node_features)
            node_embeddings = self.model(g, self._to_tensor(dict(g.ndata)))

        if len(node_embeddings.shape) == 3:
            # -> (nodes*batch_size, out_node_features)
            return th.mean(node_embeddings, dim=1)

        #  -> (nodes*batch_size, out_node_features)
        return node_embeddings

    def get_embedding_dimension(self) -> int:
        return self.architecture.layers[-1].kwargs["out_feats"]

    def save(self) -> None:
        if self.log_file is None:
            raise Exception("Called save() in " + self.name + " with None log_dir")

        checkpoint = {
            "name": self.name,
            "network_state": self.state_dict(),
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "architecture": asdict(self.architecture),
        }

        th.save(checkpoint, self.log_file)

    @staticmethod
    def load(log_file: str) -> "GCN":
        checkpoint = th.load(log_file)
        gcn = GCN(
            node_features=checkpoint["node_features"],
            edge_features=checkpoint["edge_features"],
            architecture=NetworkArchitecture(**checkpoint["architecture"]),
            name=checkpoint["name"],
            log_dir=str(Path(log_file).parents[0]),
        )

        gcn.load_state_dict(checkpoint["network_state"])
        return gcn

    @staticmethod
    def _to_tensor(d: Dict[Any, Tensor]) -> Tensor:
        features: List[Tensor] = list(d.values())
        if features:
            return th.stack(features).transpose(0, 1).float()
        return th.empty().float()

    @staticmethod
    def _add_self_loop_to_batched_graph(g: DGLHeteroGraph) -> DGLHeteroGraph:
        num_nodes = g.batch_num_nodes()
        num_edges = g.batch_num_edges()
        g = dgl.add_self_loop(g)
        g.set_batch_num_nodes(num_nodes)
        g.set_batch_num_edges(num_edges)
        return g
