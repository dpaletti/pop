from dataclasses import dataclass
from typing import Optional

import toml
from configs.agent_architecture import AgentArchitecture
from configs.type_aliases import ParsedTOMLDict


@dataclass(frozen=True)
class POPArchitecture:
    agent_neighbourhood_radius: int
    decentralized: bool
    epsilon_beta_scheduling: bool
    enable_power_supply_modularity: bool
    manager_history_size: int
    agent_type: str


@dataclass(frozen=True)
class Architecture:
    pop: POPArchitecture

    agent: AgentArchitecture

    manager: AgentArchitecture

    head_manager: AgentArchitecture

    def __init__(
        self,
        path: Optional[str] = None,
        network_architecture_implementation_folder_path: Optional[str] = None,
        network_architecture_frame_folder_path: Optional[str] = None,
        load_from_dict: Optional[dict] = None,
    ):
        if load_from_dict is not None:
            object.__setattr__(self, "pop", POPArchitecture(**load_from_dict["pop"]))
            object.__setattr__(
                self, "agent", AgentArchitecture(load_from_dict=load_from_dict["agent"])
            )
            object.__setattr__(
                self,
                "manager",
                AgentArchitecture(load_from_dict=load_from_dict["manager"]),
            )
            object.__setattr__(
                self,
                "head_manager",
                AgentArchitecture(load_from_dict=load_from_dict["head_manager"]),
            )
            return
        architecture_dict: ParsedTOMLDict = toml.load(open(path))

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
