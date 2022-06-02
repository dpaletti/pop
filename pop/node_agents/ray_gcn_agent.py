from typing import Union, Optional

import ray

from node_agents.base_gcn_agent import BaseGCNAgent
import torch as th


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
        return [
            self.optimizer.state_dict(),
            self.q_network.state_dict(),
            self.target_network.state_dict(),
            self.losses,
            self.actions_taken,
            self.decay_steps,
            self.alive_steps,
            self.trainsteps,
            self.learning_steps,
        ]

    def get_name(self):
        return self.name

    def load_state(
        self,
        optimizer_state,
        q_network_state,
        target_network_state,
        losses,
        actions,
        decay_steps,
        alive_steps,
        trainsteps,
        learning_steps,
        reset_decay=False,
    ):
        self.optimizer.load_state_dict(optimizer_state)
        self.q_network.load_state_dict(q_network_state)
        self.target_network.load_state_dict(target_network_state)
        self.losses = losses
        self.actions_taken = actions
        self.decay_steps = decay_steps if not reset_decay else 0
        self.alive_steps = alive_steps
        self.trainsteps = trainsteps
        self.learning_steps = learning_steps
