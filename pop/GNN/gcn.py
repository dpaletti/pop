import json
from abc import abstractmethod
from typing import Any, Union
import dgl

import torch as th
import torch.nn as nn
from pathlib import Path

from dgl import DGLHeteroGraph
from prettytable import PrettyTable
from torch import Tensor


class GCN(nn.Module):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        architecture_path: str,
        name: str,
        log_dir: str = "./",
        **kwargs,
    ) -> None:
        super(GCN, self).__init__()

        # Retrieving architecture from JSON
        self.name: str = name
        self.architecture: dict[str, Any] = self.load_architecture(architecture_path)

        # Logging path
        self.log_dir: str = log_dir
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = str(Path(self.log_dir, self.name + ".pt"))

        # Logging
        self.log_file: str = log_dir
        self.name: str = name
        Path(self.log_file).mkdir(parents=True, exist_ok=True)
        self.log_file = str(Path(self.log_file, name + ".pt"))

        # Parameters
        self.node_features = node_features
        self.edge_features = edge_features

    def save(self):
        checkpoint = {
            "name": self.name,
            "network_state": self.state_dict(),
            "node_features": self.node_features,
            "edge_features": self.edge_features,
        }
        checkpoint = checkpoint | self.architecture
        th.save(checkpoint, self.log_file)

    def load(self, log_dir: str):
        checkpoint = th.load(log_dir)
        self.name = checkpoint["name"]
        self.load_state_dict(checkpoint["network_state"])
        self.node_features = checkpoint["node_features"]
        self.edge_features = checkpoint["edge_features"]
        self.architecture = {
            i: j
            for i, j in checkpoint.items()
            if i
            not in {
                "network_state",
                "node_features",
                "edge_features",
                "name",
            }
        }
        print("Network Succesfully Loaded!")

    def load_architecture(self, architecture: Union[str, dict]) -> dict:
        try:
            if type(architecture) != dict:
                architecture = json.load(open(architecture))
                print(
                    "Architecture succesfully loaded for "
                    + self.name
                    + " from "
                    + architecture
                )
            return architecture
        except Exception as e:
            raise Exception(
                "Could not open architecture json at "
                + architecture
                + "\n encountered exception:\n"
                + str(e)
            )

    def get_embedding_dimension(self):
        raise Exception("get_embedding_dimension is not implemented for " + self.name)

    @staticmethod
    def count_parameters(model):
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        non_trainable_params = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                params = parameter.numel()
                table.add_row([name + " (non trainable)", params])
                non_trainable_params += params
            else:
                params = parameter.numel()
                table.add_row([name, params])
                total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")
        print(f"Total Non Trainable Params: {non_trainable_params}")
        return total_params, non_trainable_params

    @staticmethod
    def dict_to_tensor(d: dict) -> Tensor:
        """
        Convert node/edge features represented as a dict from Deep Graph Library
        to a tensor

        Parameters
        ----------
        d: ``dict``
            input dictionary with keys as feature names and values as feature values
        """

        return (
            th.stack([column.data for column in list(d.values())])
            .transpose(0, 1)
            .float()
        )

    @staticmethod
    def preprocess_graph(g: DGLHeteroGraph) -> DGLHeteroGraph:
        num_nodes = g.batch_num_nodes()
        num_edges = g.batch_num_edges()
        g = dgl.add_self_loop(g)
        g.set_batch_num_nodes(num_nodes)
        g.set_batch_num_edges(num_edges)
        return g
