from typing import Union, Tuple, List, OrderedDict

import ray
from torch import Tensor

from networks.dueling_net import DuelingNet
from node_agents.base_gcn_agent import BaseGCNAgent

from node_agents.ray_agent import RayAgent


@ray.remote
class RayGCNAgent(BaseGCNAgent, RayAgent):
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

    def get_q_network(self) -> DuelingNet:
        return self.q_network

    def get_state(
        self,
    ) -> Tuple[
        dict,
        OrderedDict[str, Tensor],
        OrderedDict[str, Tensor],
        List[float],
        List[int],
        int,
        int,
        int,
        int,
    ]:
        return (
            self.optimizer.state_dict(),
            self.q_network.state_dict(),
            self.target_network.state_dict(),
            self.losses,
            self.actions_taken,
            self.decay_steps,
            self.alive_steps,
            self.trainsteps,
            self.learning_steps,
        )

    def get_name(self) -> str:
        return self.name

    def load_state(
        self,
        optimizer_state: dict,
        q_network_state: OrderedDict[str, Tensor],
        target_network_state: OrderedDict[str, Tensor],
        losses: List[float],
        actions: List[int],
        decay_steps: int,
        alive_steps: int,
        trainsteps: int,
        learning_steps: int,
        reset_decay=False,
    ) -> None:
        self.optimizer.load_state_dict(optimizer_state)
        self.q_network.load_state_dict(q_network_state)
        self.target_network.load_state_dict(target_network_state)
        self.losses = losses
        self.actions_taken = actions
        self.decay_steps = decay_steps if not reset_decay else 0
        self.alive_steps = alive_steps
        self.trainsteps = trainsteps
        self.learning_steps = learning_steps
