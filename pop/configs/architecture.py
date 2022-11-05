from dataclasses import dataclass
from typing import Dict, List, Optional, Any

import toml

from pop.configs.agent_architecture import AgentArchitecture
from pop.configs.type_aliases import ParsedTOMLDict


@dataclass(frozen=True)
class POPArchitecture:
    node_features: List[str]
    edge_features: List[str]
    agent_neighbourhood_radius: int = 1
    head_manager_embedding_name: str = "embedding_community_action"
    decentralized: bool = False
    epsilon_beta_scheduling: bool = False
    enable_power_supply_modularity: bool = False
    manager_history_size: int = int(1e5)
    manager_initialization_half_life: int = 0
    agent_type: str = "uniform"
    disabled_action_loops_length: int = 0
    repeated_action_penalty: float = 0
    manager_selective_learning: bool = False
    agent_selective_learning: bool = False
    composite_actions: bool = False
    no_action_reward: bool = False
    incentives: Optional[Dict[str, Any]] = None
    dictatorship_penalty: Optional[Dict[str, Any]] = None
    enable_expert: bool = False
    safe_max_rho: float = 0.99
    curtail_storage_limit: float = 10
    actions_per_generator: int = 10
    generator_storage_only: bool = False
    remove_no_action: bool = False
    manager_remove_no_action: bool = False


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
