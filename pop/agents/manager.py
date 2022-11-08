import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple
from pop.configs import architecture

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

from pop.constants import PER_PROCESS_GPU_MEMORY_FRACTION

logging.getLogger("lightning").addHandler(logging.NullHandler())
logging.getLogger("lightning").propagate = False

warnings.filterwarnings("ignore", category=UserWarning)


@ray.remote(
    num_gpus=0 if not th.cuda.is_available() else PER_PROCESS_GPU_MEMORY_FRACTION,
)
class Manager(BaseGCNAgent):
    def __init__(
        self,
        agent_actions: int,
        node_features: List[str],
        architecture: AgentArchitecture,
        name: str,
        training: bool,
        device: str,
        feature_ranges: Dict[str, Tuple[float, float]],
        edge_features: Optional[List[str]] = None,
        single_node_features: Optional[int] = None,
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
            single_node_features=single_node_features,
            feature_ranges=feature_ranges,
        )

        self.embedding_size = self.q_network.get_embedding_size()

    def get_embedding_size(self) -> int:
        return self.embedding_size

    def get_node_embeddings(self, transformed_observation: DGLHeteroGraph) -> Tensor:
        if self.edge_features is not None:
            if transformed_observation.num_edges() == 0:
                transformed_observation.add_edge([0], [0])
                self._add_fake_edge_features(transformed_observation)
        return self.q_network.embedding(transformed_observation.to(self.device))

    def _take_action(
        self, transformed_observation: DGLHeteroGraph, mask: List[int] = None
    ) -> int:
        # Mask must be not none

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
            architecture=AgentArchitecture(load_from_dict=checkpoint["architecture"])
            if kwargs.get("architecture") is None
            else kwargs["architecture"],
            name=checkpoint["name"],
            training=kwargs["training"],
            device=checkpoint["device"],
            edge_features=checkpoint["edge_features"],
            single_node_features=checkpoint["single_node_features"],
            feature_ranges=checkpoint["feature_ranges"],
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
            reset_exploration=kwargs["reset_exploration"],
        )
        return manager
