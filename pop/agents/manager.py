from dataclasses import asdict
from typing import Optional, List, Dict, Any, OrderedDict

from ray import ObjectRef
from torch import Tensor

from agents.base_gcn_agent import BaseGCNAgent
from configs.agent_architecture import AgentArchitecture
from dgl import DGLHeteroGraph
import numpy as np
import torch as th
import ray


@ray.remote
class Manager(BaseGCNAgent):
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

        self.node_embeddings: Optional[Tensor] = None
        self.embedding_size = self.q_network.get_embedding_size()

    def get_embedding_size(self):
        return self.embedding_size

    def get_node_embeddings(self) -> Tensor:
        if self.node_embeddings is None:
            raise Exception("None embedding in: " + self.name)
        return self.node_embeddings

    def take_action(
        self, transformed_observation: DGLHeteroGraph, mask: Optional[List[int]] = None
    ) -> int:
        if mask is None:
            raise Exception(
                "Manager '" + self.name + " take_action() invoked with None mask"
            )

        action_list = list(range(self.actions))

        self.epsilon = self._exponential_decay(
            self.architecture.exploration.max_epsilon,
            self.architecture.exploration.min_epsilon,
            self.architecture.exploration.epsilon_decay,
        )

        self.node_embeddings = self.q_network.embedding(transformed_observation)

        if self.training:
            # epsilon-greedy Exploration
            if np.random.rand() <= self.epsilon:
                return np.random.choice(
                    action_list,
                    p=[
                        1 / len(mask) if action in mask else 0 for action in action_list
                    ],
                )

        # -> (actions)
        advantages: Tensor = self.q_network.advantage(transformed_observation)

        # Masking advantages
        advantages[
            [False if action in mask else True for action in action_list]
        ] = float("-inf")

        return int(th.argmax(advantages).item())

    @staticmethod
    def factory(checkpoint: Dict[str, Any], **kwargs) -> ObjectRef:
        manager: ObjectRef = Manager.remote(
            agent_actions=checkpoint["agent_actions"],
            node_features=checkpoint["node_features"],
            architecture=AgentArchitecture(load_from_dict=checkpoint["architecture"]),
            name=checkpoint["name"],
            training=checkpoint["training"],
            device=checkpoint["device"],
            edge_features=checkpoint["edge_features"],
        )
        manager.load_state.remote(
            optimizer_state=checkpoint["optimizer_state"],
            q_network_state=checkpoint["q_network_state"],
            target_network_state=checkpoint["target_network_state"],
            decay_steps=checkpoint["decay_steps"],
            alive_steps=checkpoint["alive_steps"],
            train_steps=checkpoint["train_steps"],
            learning_steps=checkpoint["learning_steps"],
        )
        return manager

    def get_state(
        self,
    ) -> Dict[str, Any]:
        return {
            "optimizer_state": self.optimizer.state_dict(),
            "q_network_state": self.q_network.state_dict(),
            "target_network_state": self.target_network.state_dict(),
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

    def load_state(
        self,
        optimizer_state: dict,
        q_network_state: OrderedDict[str, Tensor],
        target_network_state: OrderedDict[str, Tensor],
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
