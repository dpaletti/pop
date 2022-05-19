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

    def get_state(self) -> dict:
        return {
            self.name: {
                "state": self.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "chosen_actions": self.chosen_actions,
                "losses": self.losses,
            }
        }

    def load_state(self, state_dict, optimizer_state_dict, actions, losses):
        self.load_state_dict(state_dict)
        self.optimizer.load_state_dict(optimizer_state_dict)
        self.chosen_actions = actions
        self.losses = losses
