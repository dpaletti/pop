from typing import Union, Optional

from GNN.conv_dueling_gcn import ConvDuelingGCN
from GNN.egat_dueling_gcn import EgatDuelingGCN
from GNN.gat_dueling_gcn import GATDuelingGCN
from GNN.gcn import GCN
import json


def get_gcn(
    is_dueling: bool,
    node_features: int,
    edge_features: int,
    architecture: Union[str, dict],
    name: str,
    log_dir: str = "./",
    action_space_size: Optional[int] = None,
    **kwargs,
) -> GCN:

    if type(architecture) is dict:
        embedding = architecture["embedding"]
    else:
        embedding = json.load(open(architecture)).get("embedding")
    if embedding is None:
        raise Exception(
            "Please add 'embedding' in the architecture json at: " + architecture
        )
    if is_dueling:

        if action_space_size is None and kwargs.get("action_space_size") is None:
            raise Exception(
                "Please pass action_space_size keyword argument for dueling GCNs"
            )
        if kwargs.get("action_space_size") is not None:
            action_space_size = kwargs["action_space_size"]
        if embedding == "conv":
            gcn = ConvDuelingGCN
        elif embedding == "egat":
            gcn = EgatDuelingGCN
        elif embedding == "gat":
            gcn = GATDuelingGCN
        else:
            raise Exception(
                "Available Embeddings for Dueling GCNs are: conv, egat and gat"
            )
        return gcn(
            node_features,
            edge_features,
            action_space_size,
            architecture,
            name,
            log_dir,
        )
