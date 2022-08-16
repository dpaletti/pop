from typing import Optional, Any, Dict, List
import dgl

import torch as th
from pathlib import Path


from dgl import DGLHeteroGraph
from torch import Tensor

from networks.network_architecture_parsing import (
    get_network,
)
from networks.serializable_module import SerializableModule
from pop.configs.network_architecture import NetworkArchitecture
from dataclasses import asdict

# ----------------------------------------------------------------#
# This imports must be aliased this way for network instantiation
# Do not remove such aliases before changing the instantiation function

# noinspection PyUnresolvedReferences
import torch.nn as nn

# noinspection PyUnresolvedReferences
import pop.networks.custom_layers as cl

# noinspection PyUnresolvedReferences
import dgl.nn.pytorch as dgl_nn

# ----------------------------------------------------------------#


class GCN(nn.Module, SerializableModule):
    def __init__(
        self,
        node_features: int,
        architecture: NetworkArchitecture,
        name: str,
        log_dir: Optional[str] = None,
        edge_features: Optional[int] = None,
    ) -> None:
        nn.Module.__init__(self)
        SerializableModule.__init__(self, log_dir, name)

        self.name: str = name

        # Fixed Features
        self.node_features: int = node_features
        self.edge_features: Optional[int] = edge_features

        # Model instantiation
        self.model: nn.Sequential = get_network(
            self, architecture, is_graph_network=True
        )
        self.architecture: NetworkArchitecture = architecture

    def forward(self, g: DGLHeteroGraph) -> Tensor:
        g = self._add_self_loop_to_batched_graph(g)
        node_embeddings: Tensor

        if self.edge_features is not None:
            # -> (nodes*batch_size, heads, out_node_features)
            node_embeddings = self.model(
                g,
                self._to_tensor(dict(g.ndata), self.node_features),
                self._to_tensor(dict(g.edata), self.edge_features),
            )

        else:
            # -> (nodes*batch_size, heads, out_node_features)
            node_embeddings = self.model(
                g, self._to_tensor(dict(g.ndata), self.node_features)
            )

        if len(node_embeddings.shape) == 3:
            # Mean over heads if multi-headed attention
            # -> (nodes*batch_size, out_node_features)
            return th.mean(node_embeddings, dim=1)

        #  -> (nodes*batch_size, out_node_features)
        return node_embeddings

    def get_embedding_dimension(self) -> int:
        return int(self.architecture.layers[-1].kwargs["out_feats"])

    def get_state(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "network_state": self.state_dict(),
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "architecture": asdict(self.architecture),
            "log_file": self.log_file,
        }

    @staticmethod
    def factory(checkpoint: Dict[str, Any], **kwargs) -> "GCN":

        gcn: "GCN" = GCN(
            node_features=checkpoint["node_features"],
            edge_features=checkpoint["edge_features"],
            architecture=NetworkArchitecture(load_from_dict=checkpoint["architecture"]),
            name=checkpoint["name"],
            log_dir=str(Path(checkpoint["log_file"]).parents[0])
            if checkpoint["log_file"] is not None
            else None,
        )

        gcn.load_state_dict(checkpoint["network_state"])
        return gcn

    @staticmethod
    def _to_tensor(d: Dict[Any, Tensor], feature_size: int) -> Tensor:
        features: List[Tensor] = list(d.values())
        if features:
            return th.stack(features).transpose(0, 1).float()
        raise Exception("Empty dict passed to _to_tensor")

    @staticmethod
    def _add_self_loop_to_batched_graph(g: DGLHeteroGraph) -> DGLHeteroGraph:
        num_nodes = g.batch_num_nodes()
        num_edges = g.batch_num_edges()
        g = dgl.add_self_loop(g)
        g.set_batch_num_nodes(num_nodes)
        g.set_batch_num_edges(num_edges)
        return g
