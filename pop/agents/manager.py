import logging
import warnings
from typing import Any, Dict, List, Optional

import ray
import torch as th
from dgl import DGLHeteroGraph
from ray import ObjectRef
from torch import Tensor

from pop.agents.base_gcn_agent import BaseGCNAgent
from pop.configs.agent_architecture import AgentArchitecture

from dgl import DGLHeteroGraph
import torch as th
import ray
import logging
import warnings

logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

warnings.filterwarnings("ignore", category=UserWarning)


@ray.remote(num_cpus=1)
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

        self.embedding_size = self.q_network.get_embedding_size()

    def get_embedding_size(self) -> int:
        return self.embedding_size

    def get_node_embeddings(self, transformed_observation: DGLHeteroGraph) -> Tensor:
        return self.q_network.embedding(transformed_observation)

    def _take_action(
        self, transformed_observation: DGLHeteroGraph, mask: List[int] = None
    ) -> int:
        # Mask must be not None, we do not check this for performance reasons

        action_list = list(range(self.actions))

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
            exploration=checkpoint["exploration"],
            alive_steps=checkpoint["alive_steps"],
            train_steps=checkpoint["train_steps"],
            learning_steps=checkpoint["learning_steps"],
        )
        return manager
