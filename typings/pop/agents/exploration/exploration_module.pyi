"""
This type stub file was generated by pyright.
"""

import abc
import dgl
from typing import Any, Callable, Dict

class ExplorationModule(abc.ABC):
    @abc.abstractmethod
    def update(self, *args, **kwargs) -> None:
        ...
    
    def action_exploration(self, action_function: Callable) -> Callable:
        ...
    
    def compute_intrinsic_reward(self, observation: dgl.DGLHeteroGraph, next_observation: dgl.DGLHeteroGraph, action: int, done: bool) -> float:
        ...
    
    @abc.abstractmethod
    def get_state(self) -> Dict[str, Any]:
        ...
    
    @staticmethod
    @abc.abstractmethod
    def load_state(state: Dict[str, Any]): # -> None:
        ...
    
    def apply_intrinsic_reward(self, step_function: Callable) -> Callable:
        ...
    

