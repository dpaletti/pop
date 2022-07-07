from dataclasses import asdict
from typing import Union, Tuple, List, OrderedDict, Dict, Any, Optional

import ray
from torch import Tensor

from configs.agent_architecture import AgentArchitecture
from networks.dueling_net import DuelingNet
from agents.base_gcn_agent import BaseGCNAgent

from agents.ray_agent import RayAgent


@ray.remote
class RayGCNAgent(BaseGCNAgent, RayAgent):
    def __init__(
        self,
        agent_actions: int,
        node_features: int,
        architecture: AgentArchitecture,
        name: str,
        training: bool,
        device: str,
        edge_features: Optional[int] = None,
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
            tensorboard_dir=None,
            log_dir=None,
        )

        # Logging
        self.losses = []
        self.actions_taken = []

    def get_q_network(self) -> DuelingNet:
        return self.q_network

    @staticmethod
    def _factory(checkpoint: Dict[str, Any]) -> "RayGCNAgent":
        agent: "RayGCNAgent" = RayGCNAgent(
            agent_actions=checkpoint["actions"],
            node_features=checkpoint["node_features"],
            architecture=AgentArchitecture(load_from_dict=checkpoint["architecture"]),
            name=checkpoint["name"],
            training=checkpoint["training"],
            device=checkpoint["device"],
            edge_features=checkpoint["edge_features"],
        )
        agent.load_state(
            optimizer_state=checkpoint["optimizer_state"],
            q_network_state=checkpoint["q_network_state"],
            target_network_state=checkpoint["target_network_state"],
            losses=checkpoint["losses"],
            actions=checkpoint["actions"],
            decay_steps=checkpoint["decay_steps"],
            alive_steps=checkpoint["alive_steps"],
            train_steps=checkpoint["train_steps"],
            learning_steps=checkpoint["learning_steps"],
        )
        return agent

    def get_state(
        self,
    ) -> Dict[str, Any]:
        return {
            "optimizer_state": self.optimizer.state_dict(),
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
            "losses": self.losses,
            "actions": self.actions_taken,
            "decay_steps": self.decay_steps,
            "alive_steps": self.alive_steps,
            "train_steps": self.train_steps,
            "learning_steps": self.learning_steps,
            "agent_actions": self.actions,
            "node_features": self.node_features,
            "edge_features": self.edge_features,
            "architecture": asdict(self.architecture),
            "name": self.name,
            "training": self.training,
            "device": self.device,
        }

    def get_name(self) -> str:
        return self.name

    def reset_decay(self):
        self.decay_steps = 0

    def load_state(
        self,
        optimizer_state: dict,
        q_network_state: OrderedDict[str, Tensor],
        target_network_state: OrderedDict[str, Tensor],
        losses: List[float],
        actions: List[int],
        decay_steps: int,
        alive_steps: int,
        train_steps: int,
        learning_steps: int,
    ) -> None:
        self.decay_steps = decay_steps
        self.alive_steps = alive_steps
        self.train_steps = train_steps
        self.learning_steps = learning_steps
        self.optimizer.load_state_dict(optimizer_state)
        self.q_network.load_state_dict(q_network_state)
        self.target_network.load_state_dict(target_network_state)
        self.losses = losses
        self.actions_taken = actions