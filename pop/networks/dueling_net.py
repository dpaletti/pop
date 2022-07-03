from typing import Optional

import torch as th
import torch.nn as nn
import dgl
from dgl.heterograph import DGLHeteroGraph
from torch import Tensor, FloatTensor

from configs.network_architecture import NetworkArchitecture
from networks.network_architecture_parsing import get_network
from pop.networks.gcn import GCN

from utilities import get_log_file


class DuelingNet(nn.Module):
    def __init__(
        self,
        action_space_size: int,
        node_features: int,
        embedding_architecture: NetworkArchitecture,
        advantage_stream_architecture: NetworkArchitecture,
        value_stream_architecture: NetworkArchitecture,
        name: str,
        log_dir: Optional[str] = None,
        edge_features: Optional[int] = None,
    ):
        super(DuelingNet, self).__init__()
        self.name = name

        self.log_file = get_log_file(log_dir, name)

        # Parameters
        self.action_space_size = action_space_size

        # Embeddings
        self.embedding: GCN = GCN(
            node_features=node_features,
            edge_features=edge_features,
            architecture=embedding_architecture,
            name=name + "_embedding",
            log_dir=log_dir,
        )

        self.embedding_size: int = self.embedding.get_embedding_dimension()

        self.advantage_stream: nn.Sequential = get_network(
            self, advantage_stream_architecture, is_graph_network=False
        )
        self.value_stream: nn.Module = get_network(
            self, value_stream_architecture, is_graph_network=False
        )

    @staticmethod
    def _compute_graph_embedding(
        g: DGLHeteroGraph, node_embeddings: th.Tensor
    ) -> th.Tensor:

        g.ndata["node_embeddings"] = node_embeddings
        graph_embedding: Tensor = dgl.mean_nodes(g, "node_embeddings")
        del g.ndata["node_embeddings"]
        return graph_embedding

    def _extract_features(self, g: DGLHeteroGraph) -> Tensor:
        node_embeddings = self.embedding(g)

        graph_embedding = self._compute_graph_embedding(g, node_embeddings)

        graph_embedding = th.flatten(graph_embedding, 1)

        return graph_embedding

    def forward(self, g: DGLHeteroGraph) -> Tensor:
        graph_embedding: Tensor = self._extract_features(g)

        # Compute value of current state
        # -> (batch_size, 1)
        state_value: float = self.value_stream(graph_embedding)

        # Compute advantage of (current_state, action) for each action
        # -> (batch_size, action_space_size)
        state_advantages: FloatTensor = self.advantage_stream(graph_embedding)

        # -> (batch_size, action_space_size)
        q_values: Tensor = state_value + (state_advantages - state_advantages.mean())

        if len(q_values.shape) == 2:
            # -> (action_space_size)
            q_values: Tensor = th.mean(q_values, 0)

        # -> (action_space_size)
        return q_values

    def advantage(self, g: DGLHeteroGraph) -> Tensor:

        graph_embedding: Tensor = self._extract_features(g)

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
