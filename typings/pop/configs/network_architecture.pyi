"""
This type stub file was generated by pyright.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
from pop.configs.type_aliases import ParsedTOMLDict

@dataclass(frozen=True)
class NetworkLayer:
    name: str
    type: str
    module: str
    kwargs: Dict[str, Union[int, float, bool, str]]
    ...


@dataclass(frozen=True)
class NetworkArchitecture:
    layers: List[NetworkLayer]
    def __init__(self, load_from_dict: Optional[Dict[str, List[ParsedTOMLDict]]] = ..., network: Optional[str] = ..., implementation_folder_path: Optional[str] = ..., frame_folder_path: Optional[str] = ...) -> None:
        ...
    


