from abc import abstractmethod
from typing import Union
import torch as th
import torch.nn as nn
import dgl
from dgl.heterograph import DGLHeteroGraph
from torch import Tensor, FloatTensor
import json
from pathlib import Path

from pop.graph_convolutional_networks.gcn import GCN
from ray.rllib.models.torch.modules.noisy_layer import NoisyLayer


class DuelingNet(nn.Module):
    def __init__(
        self,
        action_space_size: int,
        architecture: Union[dict, str],
        name: str,
        log_dir: str,
    ):
        super(DuelingNet, self).__init__()
        self.name = name

        self.log_dir = log_dir
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.log_file = str(Path(self.log_dir, self.name + ".pt"))

        self.architecture = (
            json.load(open(architecture))
            if type(architecture) is not dict
            else architecture
        )

        # Parameters
        self.action_space_size = action_space_size

        # Network Paths
        self.advantage_stream: nn.Module = self.init_advantage_stream(action_space_size)
        self.value_stream: nn.Module = self.init_value_stream()

    @property
    @abstractmethod
    def embedding(self) -> GCN:
        ...

    @abstractmethod
    def extract_features(self, g: DGLHeteroGraph):
        ...

    def init_advantage_stream(self, action_space_size: int) -> nn.Module:
        if not self.architecture.get("noisy_layers"):
            return nn.Sequential(
                nn.Linear(
                    self.architecture["hidden_output_size"]
                    * self.architecture["heads"][-1],
                    self.architecture["advantage_stream_size"],
                ),
                nn.ReLU(),
                nn.Linear(
                    self.architecture["advantage_stream_size"], action_space_size
                ),
            )
        else:
            # Activation is already inside the noisy layer and it is already ReLU
            return nn.Sequential(
                NoisyLayer(
                    self.architecture["hidden_output_size"]
                    * self.architecture["heads"][-1],
                    self.architecture["advantage_stream_size"],
                    sigma0=0.5,
                ),
                NoisyLayer(
                    self.architecture["advantage_stream_size"],
                    action_space_size,
                    sigma0=0.5,
                ),
            )

    def init_value_stream(self) -> nn.Module:
        if not self.architecture.get("noisy_layers"):
            return nn.Sequential(
                nn.Linear(
                    self.architecture["hidden_output_size"]
                    * self.architecture["heads"][-1],
                    self.architecture["value_stream_size"],
                ),
                nn.ReLU(),
                nn.Linear(self.architecture["value_stream_size"], 1),
            )
        else:
            return nn.Sequential(
                NoisyLayer(
                    self.architecture["hidden_output_size"]
                    * self.architecture["heads"][-1],
                    self.architecture["value_stream_size"],
                    sigma0=0.5,
                ),
                NoisyLayer(self.architecture["value_stream_size"], 1, sigma0=0.5),
            )

    @staticmethod
    def compute_graph_embedding(
        g: DGLHeteroGraph, node_embeddings: th.Tensor
    ) -> th.Tensor:

        g.ndata["node_embeddings"] = node_embeddings
        graph_embedding: Tensor = dgl.mean_nodes(g, "node_embeddings")
        del g.ndata["node_embeddings"]
        return graph_embedding

    def forward(self, g: DGLHeteroGraph) -> Tensor:
        graph_embedding: Tensor = self.extract_features(g)

        # Compute value of current state
        state_value: float = self.value_stream(graph_embedding)

        # Compute advantage of (current_state, action) for each action
        state_advantages: FloatTensor = self.advantage_stream(graph_embedding)

        q_values: Tensor = state_value + (state_advantages - state_advantages.mean())
        return q_values

    def advantage(self, g: DGLHeteroGraph) -> Tensor:
        """
        Advantage value for each state-action pair

        Parameters
        ----------
        g: :class:`DGLHeteroGraph`
            input graph

        Return
        ------
        state_advantages: :class: `Tensor`
            state-action advantages

        """

        graph_embedding: Tensor = self.extract_features(g)

        # Compute advantage of (current_state, action) for each action
        state_advantages: FloatTensor = self.advantage_stream(graph_embedding)

        return state_advantages

    def save(self):
        """
        Save a checkpoint of the network.
        Save all the hyperparameters.
        """
        checkpoint = {
            "name": self.name,
            "network_state": self.state_dict(),
            "action_space_size": self.action_space_size,
        }
        checkpoint = dict(list(checkpoint.items()) + list(self.architecture.items()))
        th.save(checkpoint, self.log_file)

    def load(
        self,
        log_dir: str,
    ):
        checkpoint = th.load(log_dir)
        self.name = checkpoint["name"]
        self.load_state_dict(checkpoint["network_state"])
        self.action_space_size = checkpoint["action_space_size"]
        self.architecture = {
            i: j
            for i, j in checkpoint.items()
            if i
            not in {
                "network_state",
                "node_features",
                "edge_features",
                "action_space_size",
                "name",
            }
        }
        print("Network Succesfully Loaded!")
