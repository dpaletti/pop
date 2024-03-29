"""
This type stub file was generated by pyright.
"""

import torch.nn as nn
from typing import Any, Dict, Optional
from dgl.heterograph import DGLHeteroGraph
from torch import Tensor
from pop.configs.network_architecture import NetworkArchitecture
from pop.networks.serializable_module import SerializableModule, T

class DuelingNet(nn.Module, SerializableModule):
    def __init__(self, action_space_size: int, node_features: int, embedding_architecture: NetworkArchitecture, advantage_stream_architecture: NetworkArchitecture, value_stream_architecture: NetworkArchitecture, name: str, log_dir: Optional[str] = ..., edge_features: Optional[int] = ...) -> None:
        ...
    
    def get_embedding_size(self) -> int:
        ...
    
    def forward(self, g: DGLHeteroGraph) -> Tensor:
        ...
    
    def advantage(self, g: DGLHeteroGraph) -> Tensor:
        ...
    
    def get_state(self: T) -> Dict[str, Any]:
        ...
    
    @staticmethod
    def factory(checkpoint: Dict[str, Any], **kwargs) -> DuelingNet:
        ...
    


