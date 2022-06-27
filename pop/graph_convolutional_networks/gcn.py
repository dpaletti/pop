from abc import ABC
from typing import Optional, Any, Dict, List
import dgl

import torch as th
import torch.nn as nn
from pathlib import Path

from dgl import DGLHeteroGraph  # type: ignore
from torch import Tensor
from pop.architectures.gcn_architecture import GCNArchitecture

# TODO: finish the architecture abstraction for GAT (needs a mean probably) and keep cleaning up stuff (e.g. return_mean_over_heads)
# TODO: remove return_mean_over_heads (is useless)


class GCN(nn.Module, ABC):
    def __init__(
        self,
        node_features: int,
        architecture: GCNArchitecture,
        name: str,
        log_dir: Optional[str] = None,
        edge_features: Optional[int] = None,
    ) -> None:
        super(GCN, self).__init__()

        self.name: str = name
        self.architecture: GCNArchitecture = architecture

        # Logging
        self.log_file: Optional[str] = None
        if log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.log_file = str(Path(log_dir, name + ".pt"))

        # Parameters
        self.node_features: int = node_features
        self.edge_features: Optional[int] = edge_features

    def save(self) -> None:
        if self.log_file is None:
            raise Exception("Called save() in " + self.name + " with None log_dir")

        checkpoint = {
            "name": self.name,
            "network_state": self.state_dict(),
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "architecture": self.architecture.dict(),
        }

        th.save(checkpoint, self.log_file)

    @classmethod
    def load(cls, log_file: str) -> "GCN":
        checkpoint = th.load(log_file)
        gcn = cls(
            node_features=checkpoint["node_features"],
            edge_features=checkpoint["edge_features"],
            architecture=GCNArchitecture.parse_obj(checkpoint["architecture"]),
            name=checkpoint["name"],
            log_dir=str(Path(log_file).parents[0]),
        )

        gcn.load_state_dict(checkpoint["network_state"])
        return gcn

    def get_embedding_dimension(self) -> Optional[int]:
        raise Exception("get_embedding_dimension is not implemented for " + self.name)

    @staticmethod
    def to_tensor(d: Dict[Any, Tensor]) -> Tensor:
        features: List[Tensor] = list(d.values())
        if features:
            return th.stack(features).transpose(0, 1).float()
        return th.empty().float()

    @staticmethod
    def add_self_loop_to_batched_graph(g: DGLHeteroGraph) -> DGLHeteroGraph:
        num_nodes = g.batch_num_nodes()
        num_edges = g.batch_num_edges()
        g = dgl.add_self_loop(g)
        g.set_batch_num_nodes(num_nodes)
        g.set_batch_num_edges(num_edges)
        return g
