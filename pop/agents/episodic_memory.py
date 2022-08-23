from networks.gcn import GCN


class EpisodicMemory:
    # TODO: choose architecture
    def __init__(
        self, node_features: int, edge_features: int, architecture: ..., name: str
    ):
        self.inverse_model = GCN(
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture,
            name=name + "_inverse_model",
            log_dir=None,
        )
