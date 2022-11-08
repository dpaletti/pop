from typing import Optional, Dict, Any, List, Tuple

import networkx as nx
from dgl import DGLHeteroGraph
from ray import ObjectRef

from pop.agents.base_gcn_agent import BaseGCNAgent
from pop.configs.agent_architecture import AgentArchitecture


class RayShallowGCNAgent(BaseGCNAgent):
    def __init__(
        self,
        name: str,
        device: str,
        agent_actions: Optional[int] = None,
        node_features: Optional[List[str]] = None,
        edge_features: Optional[List[str]] = None,
        architecture: Optional[AgentArchitecture] = None,
        training: bool = False,
    ):
        self.name = name
        self.device = device

    def get_state(
        self,
    ) -> Dict[str, Any]:
        return {
            "name": self.name,
            "device": self.device,
        }

    def get_exploration_logs(self) -> Dict[str, Any]:
        return {}

    def get_name(self) -> str:
        return self.name

    @staticmethod
    def factory(checkpoint: Dict[str, Any], **kwargs):
        agent = RayShallowGCNAgent(name=checkpoint["name"], device=checkpoint["device"])
        return agent

    def take_action(
        self, transformed_observation: DGLHeteroGraph, mask: List[int] = None
    ) -> Tuple[int, float]:
        return 0, 0  # Always no-action with 0 q-value

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
