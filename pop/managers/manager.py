import json
from abc import abstractmethod
from typing import Union, Optional
from pathlib import Path

import torch.nn as nn
import torch as th

from graph_convolutional_networks.gcn import GCN
from managers.node_attention import NodeAttention


class Manager(nn.Module):
    def __init__(
        self,
        node_features: int,
        edge_features: Optional[int],
        architecture: Union[str, dict],
        name: str,
        log_dir: str,
    ):
        super(Manager, self).__init__()

        self.node_features = node_features
        self.edge_features = edge_features
        self.architecture = (
            json.load(open(architecture))
            if type(architecture) is not dict
            else architecture
        )
        self.name = name
        self.log_dir = log_dir
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = str(Path(self.log_dir, name + ".pt"))

    @property
    @abstractmethod
    def node_attention(self) -> NodeAttention:
        ...

    @property
    @abstractmethod
    def embedding(self) -> GCN:
        ...

    def save(self):
        checkpoint = {
            "name": self.name,
            "manager_state": self.state_dict(),
            "node_features": self.node_features,
            "edge_features": self.edge_features,
        }
        checkpoint = checkpoint | self.architecture
        print("Saving checkpoint to: " + str(self.log_file))
        th.save(checkpoint, self.log_file)

    @classmethod
    def load(cls, log_file: str):
        checkpoint = th.load(log_file)
        architecture = {
            i: j
            for i, j in checkpoint.items()
            if i not in {"manager_state", "node_features", "edge_features", "name"}
        }
        print("Manager Succesfully Loaded!")
        manager = cls(
            node_features=checkpoint["node_features"],
            edge_features=checkpoint["edge_features"],
            architecture=architecture,
            name=checkpoint["name"],
            log_dir=Path(log_file).parents[0],
        )
        manager.load_state_dict(checkpoint["manager_state"])
        return manager
