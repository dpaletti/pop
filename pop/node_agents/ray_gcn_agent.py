from typing import Union

import ray

from node_agents.base_gcn_agent import BaseGCNAgent


@ray.remote
class RayGCNAgent(BaseGCNAgent):
    def __init__(
        self,
        agent_actions: int,
        node_features: int,
        edge_features: int,
        architecture: Union[str, dict],
        name: str,
        training: bool,
        device: str,
    ):
        BaseGCNAgent.__init__(
            self,
            agent_actions=agent_actions,
            node_features=node_features,
            edge_features=edge_features,
            architecture=architecture,
            name=name,
            training=training,
            device=device,
        )

        # Logging
        self.losses = []
        self.actions_taken = []

    def get_state(self):
        return {
            self.name: {
                "optimizer_state": self.optimizer.state_dict(),
                "q_network_state": self.q_network.state_dict(),
                "target_network_state": self.target_network.state_dict(),
                "losses": self.losses,
                "actions": self.actions_taken,
            }
        }

    def load_state(
        self, optimizer_state, q_network_state, target_network_state, losses, actions
    ):
        self.optimizer.load_state_dict(optimizer_state)
        self.q_network.load_state_dict(q_network_state)
        self.target_network.load_state_dict(target_network_state)
        self.losses = losses
        self.actions_taken = actions
