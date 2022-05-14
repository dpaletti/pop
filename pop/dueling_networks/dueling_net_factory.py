import json
from typing import Union, Optional

from dueling_networks.dueling_net import DuelingNet
from dueling_networks.egat_dueling_gcn import EgatDuelingGCN
from dueling_networks.gat_dueling_gcn import GatDuelingGCN


def get_dueling_net(
    name: str,
    architecture: Union[str, dict],
    node_features: int,
    action_space_size: int,
    edge_features: Optional[int] = None,
    log_dir: str = "./",
) -> DuelingNet:

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
        return EgatDuelingGCN(
            node_features, edge_features, action_space_size, architecture, name, log_dir
        )

    if embedding == "gat":
        if edge_features is not None:
            print("WARNING: Edge features are ignored by GAT embedding")
        return GatDuelingGCN(
            node_features, action_space_size, architecture, name, log_dir
        )

    raise Exception(
        "Embedding: " + str(embedding) + " not among the available ones: egat, gat"
    )
