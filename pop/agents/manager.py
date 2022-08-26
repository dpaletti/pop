from typing import Optional, List, Dict, Any

from ray import ObjectRef
from torch import Tensor

from pop.agents.base_gcn_agent import BaseGCNAgent
from pop.configs.agent_architecture import AgentArchitecture
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
        cpu_affinity: Optional[List[int]] = None,
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

        self.embedding_size = self.q_network.get_embedding_size()

    def get_embedding_size(self) -> int:
        return self.embedding_size

    def get_node_embeddings(self, transformed_observation: DGLHeteroGraph) -> Tensor:
        return self.q_network.embedding(transformed_observation)

    def take_action(
        self, transformed_observation: DGLHeteroGraph, mask: List[int] = None
    ) -> int:

        action_list = list(range(self.actions))

        self.epsilon = self._exponential_decay(
            self.architecture.exploration.max_epsilon,
            self.architecture.exploration.min_epsilon,
            self.architecture.exploration.epsilon_decay,
        )

        node_embeddings = self.get_node_embeddings(transformed_observation)

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
            memory=checkpoint["memory"],
            decay_steps=checkpoint["decay_steps"],
            alive_steps=checkpoint["alive_steps"],
            train_steps=checkpoint["train_steps"],
            learning_steps=checkpoint["learning_steps"],
        )
        return manager
