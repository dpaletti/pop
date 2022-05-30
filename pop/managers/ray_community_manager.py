from typing import Union, List

import ray

from managers.community_manager import CommunityManager

import torch as th


@ray.remote
class RayCommunityManager(CommunityManager):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        architecture: Union[str, dict],
        name: str,
    ):
        CommunityManager.__init__(
            self,
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture,
            name=name,
            log_dir=None,
        )

        # Logging
        self.chosen_actions: List[int] = []
        self.losses: List[float] = []

        # Optimizer
        self.optimizer = th.optim.Adam(
            self.parameters(), lr=self.architecture["learning_rate"]
        )

    def learn(self, loss: float):
        loss = th.Tensor(loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.data)

    def get_state(self):
        return [
            self.embedding.state_dict(),
            self.node_attention.state_dict(),
            self.optimizer.state_dict(),
            self.chosen_actions,
            self.losses,
        ]

    def get_name(self):
        return self.name

    def load_state(
        self,
        embedding_state_dict,
        node_attention_state_dict,
        optimizer_state_dict,
        actions,
        losses,
    ):
        self.embedding.load_state_dict(embedding_state_dict)
        self.node_attention.state_dict(node_attention_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.chosen_actions = actions
        self.losses = losses