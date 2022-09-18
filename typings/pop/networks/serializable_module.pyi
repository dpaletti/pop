"""
This type stub file was generated by pyright.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypeVar

T = TypeVar("T")
class SerializableModule(ABC):
    def __init__(self, log_dir: Optional[str], name: Optional[str]) -> None:
        ...
    
    @abstractmethod
    def get_state(self: T) -> Dict[str, Any]:
        ...
    
    def save(self: T) -> None:
        ...
    
    @classmethod
    def load(cls, log_file: Optional[str] = ..., checkpoint: Optional[Dict[str, Any]] = ..., **kwargs) -> T:
        ...
    
    @staticmethod
    @abstractmethod
    def factory(checkpoint: Dict[str, Any], **kwargs) -> T:
        ...
    

