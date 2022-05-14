from typing import Union, Optional

from graph_convolutional_networks.egat_gcn import EgatGCN
from graph_convolutional_networks.gat_gcn import GatGCN
from graph_convolutional_networks.gcn import GCN
import json


def get_gcn(
    name: str,
    architecture: Union[str, dict],
    node_features: int,
    edge_features: Optional[int] = None,
    log_dir: str = "./",
) -> GCN:

    if type(architecture) is dict:
        embedding = architecture["embedding"]
    else:
        embedding = json.load(open(architecture)).get("embedding")

    if embedding is None:
        raise Exception(
            "Please add 'embedding' in the architecture json at: " + architecture
        )

    if embedding == "egat":
        if edge_features is None:
            raise Exception("Please pass edge features for EGAT embedding")
        return EgatGCN(node_features, edge_features, architecture, name, log_dir)

    if embedding == "gat":
        if edge_features is not None:
            print("WARNING: Edge features are ignored by GAT embedding")
        return GatGCN(node_features, architecture, name, log_dir)

    raise Exception(
        "Embedding: " + str(embedding) + " not among the available ones: egat, gat"
    )
