from typing import Union, List

import ray

from managers.community_manager import CommunityManager

import torch as th

from torchinfo import summary


@ray.remote
class RayCommunityManager(CommunityManager):
    def __init__(
        self,
        node_features: int,
        edge_features: int,
        architecture: Union[str, dict],
        training: bool,
        name: str,
    ):
        CommunityManager.__init__(
            self,
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture,
            name=name,
            log_dir=None,
            training=training,
        )

        # Logging
        self.chosen_actions: List[int] = []
        self.losses: List[float] = []

        # Optimizer
        self.optimizer = th.optim.Adam(
            self.parameters(), lr=self.architecture["learning_rate"]
        )

    def learn(self, reward):
        loss = (
            -self.node_choice.attention_distribution.log_prob(
                th.tensor(self.current_best_node)
            )
            * reward
        )
        self.optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(self.parameters(), self.architecture["max_clip"])
        self.optimizer.step()

        self.losses.append(loss.data)

    def get_state(self):
        return [
            self.embedding.state_dict(),
            self.node_choice.state_dict(),
            self.optimizer.state_dict(),
            self.chosen_actions,
            self.losses,
        ]

    def get_summary(self):
        return summary(self)

    def get_embedding(self):
        return self.embedding

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
        self.node_choice.state_dict(node_attention_state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.chosen_actions = actions
        self.losses = losses
