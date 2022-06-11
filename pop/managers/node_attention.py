from typing import Optional, Union

import torch.nn as nn
from torch import Tensor
import torch as th


class NodeAttention(nn.Module):
    def __init__(self, architecture: dict, embedding_dimension: int, training: bool):
        super(NodeAttention, self).__init__()
        self.architecture = architecture

        self.training = training

        self.batch_size = self.architecture["batch_size"]

        self.qkv_projection = nn.Linear(
            embedding_dimension,
            3 * self.architecture["embedding_dimension"],
            bias=False,
        ).float()

        self.attention = nn.MultiheadAttention(
            self.architecture["embedding_dimension"],
            self.architecture["heads"],
            self.architecture["dropout"],
            bias=True,
            dtype=float,
            batch_first=False,
        ).float()

        self.attention_projection = nn.Linear(
            self.architecture["embedding_dimension"], 1, bias=False
        ).float()

        if self.training:
            self.softmax = nn.Softmax(dim=0)
            self.attention_distribution: Optional[th.distributions.Distribution] = None

    def forward(self, node_embedding: Tensor) -> int:

        # -> (Nodes, 3 * Embedding Size, Batch Size)
        # Projection to the Query-Key-Value Space
        qkv = self.qkv_projection(node_embedding)

        # -> (Nodes, (if optional = 1, gets squeezed) Batch Size, 3 * Embedding Size)
        qkv = qkv.reshape(
            node_embedding.shape[0],
            self.batch_size,
            3 * self.architecture["embedding_dimension"],
        ).squeeze()

        # -> 3 * (Nodes, Batch Size, Embedding Size)
        q, k, v = qkv.chunk(3, dim=-1)

        # -> (Nodes, [Batch Size], Attention Space Size)
        # Heads are automatically averaged inside the attention module
        attention_output, _ = self.attention(q, k, v)

        # -> (Nodes, [Batch Size])
        projected_attention = self.attention_projection(attention_output).squeeze(
            dim=-1
        )

        # -> (Nodes,)
        if len(projected_attention.shape) > 1:
            mean_over_batch = th.mean(projected_attention, dim=-1)
        else:
            mean_over_batch = projected_attention

        if not self.training:
            # Not Learnable
            # -> (1,)
            best_node = th.argmax(mean_over_batch)
        else:
            # Learnable

            # -> (Nodes, )

            self.attention_distribution: th.distributions.Distribution = (
                th.distributions.Categorical(self.softmax(mean_over_batch))
            )

            # -> (1, )
            best_node = self.attention_distribution.sample()

        return int(best_node)
