from typing import Optional, Dict, Any, List, Tuple

import networkx as nx
import ray
from dgl import DGLHeteroGraph
from ray import ObjectRef

from pop.agents.base_gcn_agent import BaseGCNAgent
from pop.configs.agent_architecture import AgentArchitecture

from pop.constants import PER_PROCESS_GPU_MEMORY_FRACTION


@ray.remote(num_gpus=PER_PROCESS_GPU_MEMORY_FRACTION * 1e-2)
class RayShallowGCNAgent(BaseGCNAgent):
    def __init__(
        self,
        name: str,
        device: str,
        agent_actions: Optional[int] = None,
        node_features: Optional[int] = None,
        edge_features: Optional[int] = None,
        architecture: Optional[AgentArchitecture] = None,
        training: bool = False,
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
            log_dir=None,
            tensorboard_dir=None,
        )

    def get_state(
        self,
    ) -> Dict[str, Any]:
        return {
            "name": self.name,
            "device": self.device,
        }

    def get_name(self) -> str:
        return self.name

    @staticmethod
    def factory(checkpoint: Dict[str, Any], **kwargs) -> ObjectRef:
        agent = RayShallowGCNAgent.remote(
            name=checkpoint["name"], device=checkpoint["device"]
        )
        return agent

    def take_action(
        self, transformed_observation: DGLHeteroGraph, mask: List[int] = None
    ) -> int:
        return 0  # Always no-action

    def step(
        self,
        observation: DGLHeteroGraph,
        action: int,
        reward: float,
        next_observation: nx.Graph,
        done: bool,
        stop_decay: bool = False,
    ) -> Tuple[None, float]:
        return None, reward
