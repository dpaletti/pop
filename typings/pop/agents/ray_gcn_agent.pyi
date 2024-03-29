"""
This type stub file was generated by pyright.
"""

import ray
from typing import Any, Dict, Optional
from ray import ObjectRef
from pop.agents.base_gcn_agent import BaseGCNAgent
from pop.configs.agent_architecture import AgentArchitecture
from pop.networks.dueling_net import DuelingNet

@ray.remote(num_cpus=1)
class RayGCNAgent(BaseGCNAgent):
    def __init__(self, agent_actions: int, node_features: int, architecture: AgentArchitecture, name: str, training: bool, device: str, edge_features: Optional[int] = ...) -> None:
        ...
    
    def get_q_network(self) -> DuelingNet:
        ...
    
    def get_name(self) -> str:
        ...
    
    def reset_decay(self): # -> None:
        ...
    
    @staticmethod
    def factory(checkpoint: Dict[str, Any], **kwargs) -> ObjectRef:
        ...
    


