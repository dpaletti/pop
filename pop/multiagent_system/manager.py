from typing import Union, Tuple
import torch as th
import torch.nn as nn
import json

from dgl import DGLHeteroGraph
from torch import Tensor
from torch.nn import MultiheadAttention
from GNN.egat_gcn import EgatGCN

# TODO: for pointer nets https://ychai.uk/notes/2019/07/21/RL/DRL/Decipher-AlphaStar-on-StarCraft-II/
# Alternating Learning: chi gioca ha un learning rate più alto
# Schedule a Turni
# Fissare le comunità
class Manager(nn.Module):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        architecture: Union[str, dict],
        name: str,
        log_dir: str,
        device: str = "cpu",
    ):
        super(Manager, self).__init__()
        self.architecture = (
            json.load(open(architecture)) if type(architecture) is str else architecture
        )
        self.batch_size = self.architecture["batch_size"]
        self.embedding = EgatGCN(
            node_features,
            edge_features,
            self.architecture["embedding_architecture"],
            name,
            log_dir,
        ).float()

        self.qkv_projection = nn.Linear(
            self.embedding.get_embedding_dimension(),
            3 * self.architecture["embedding_dimension"],
            bias=False,
        ).float()

        self.attention = MultiheadAttention(
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

    def forward(self, g: DGLHeteroGraph) -> Tuple[int, DGLHeteroGraph]:
        # -> (Nodes, Heads, Embedding Size, (optional) Batch)
        # -> (Edges, Heads, Embedding, Size, (optional) Batch)
        (
            node_embedding,
            edge_embedding,
            g,
        ) = self.embedding.compute_embeddings(g, return_graph=True)

        # -> (Nodes, Embedding Size, (optional) Batch Size)
        node_embedding: Tensor = self.embedding.compute_node_embedding(
            g, node_embedding, edge_embedding
        )

        # -> (Nodes, 3 * Embedding Size, Batch Size)
        # Projection to the Query-Key-Value Space
        qkv = self.qkv_projection(node_embedding)

        # -> (Nodes, (if optional = 1, gets squeezed) Batch Size, 3 * Embedding Size)
        qkv = qkv.reshape(
            g.num_nodes(),
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
        best_node = th.argmax(mean_over_batch)

        g.ndata["embedding"] = node_embedding

        return g.nodes[best_node].data["action"], g
