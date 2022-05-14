import torch.nn as nn
from torch import Tensor
import torch as th


class NodeAttention(nn.Module):
    def __init__(
        self, architecture: dict, embedding_dimension: int, device: str = "cpu"
    ):
        super(NodeAttention, self).__init__()
        self.architecture = architecture

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
            device=device,
            dtype=float,
            batch_first=False,
        ).float()

        self.attention_projection = nn.Linear(
            self.architecture["embedding_dimension"], 1, bias=False
        ).float()

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

        # -> (Nodes, Batch Size, Attention Space Size)
        # Heads are automatically averaged inside the attention module
        attention_output, _ = self.attention(q, k, v)

        # -> (Nodes, Batch Size)
        projected_attention = self.attention_projection(attention_output).squeeze(
            dim=-1
        )

        # -> (Nodes, 1)
        mean_over_batch = th.mean(projected_attention, dim=-1)

        # -> Scalar
        best_node = int(th.argmax(mean_over_batch))

        return best_node
