from dataclasses import dataclass
from typing import Dict, Union
import toml
from configs.agent_architecture import AgentArchitecture


@dataclass(frozen=True)
class POPArchitecture:
    agent_neighbourhood_radius: int
    decentralized: bool
    fixed_communities: bool
    epsilon_beta_scheduling: bool
    agent_type: str


@dataclass(frozen=True)
class Architecture:
    pop: POPArchitecture

    agent: AgentArchitecture

    manager: AgentArchitecture

    head_manager: AgentArchitecture

    def __init__(
        self,
        path: str,
        network_architecture_implementation_folder_path: str,
        network_architecture_frame_folder_path: str,
    ):
        architecture_dict: Dict[
            str, Dict[str, Union[int, bool, str, float]]
        ] = toml.load(open(path))

        assert "pop" in architecture_dict.keys()
        assert "agent" in architecture_dict.keys()
        assert "manager" in architecture_dict.keys()
        assert "head_manager" in architecture_dict.keys()

        object.__setattr__(self, "pop", POPArchitecture(**architecture_dict["pop"]))
        object.__setattr__(
            self,
            "agent",
            AgentArchitecture(
                architecture_dict["agent"],
                network_architecture_implementation_folder_path=network_architecture_implementation_folder_path,
                network_architecture_frame_folder_path=network_architecture_frame_folder_path,
            ),
        )
        object.__setattr__(
            self,
            "manager",
            AgentArchitecture(
                architecture_dict["manager"],
                network_architecture_implementation_folder_path=network_architecture_implementation_folder_path,
                network_architecture_frame_folder_path=network_architecture_frame_folder_path,
            ),
        )
        object.__setattr__(
            self,
            "head_manager",
            AgentArchitecture(
                architecture_dict["head_manager"],
                network_architecture_implementation_folder_path=network_architecture_implementation_folder_path,
                network_architecture_frame_folder_path=network_architecture_frame_folder_path,
            ),
        )
