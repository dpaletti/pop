import json
from abc import abstractmethod
from typing import Any

import torch as th
import torch.nn as nn
from pathlib import Path
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

    @staticmethod
    def get_GCN(
        node_features: int,
        edge_features: int,
        architecture_path: str,
        name: str,
        log_dir: str = "./",
        **kwargs,
    ):
        ...

    @abstractmethod
    def save(self):
        ...

    @abstractmethod
    def load(self, log_dir: str):
        ...

    def load_architecture(self, architecture_path: str) -> dict:
        try:
            architecture_dict = json.load(open(architecture_path))
            print(
                "Architecture succesfully loaded for "
                + self.name
                + " from "
                + architecture_path
            )
            return architecture_dict
        except Exception as e:
            raise Exception(
                "Could not open architecture json at "
                + architecture_path
                + "\n encountered exception:\n"
                + str(e)
            )

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
