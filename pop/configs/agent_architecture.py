from dataclasses import dataclass
from typing import Optional

from configs.network_architecture import NetworkArchitecture
from configs.type_aliases import EventuallyNestedDict


@dataclass(frozen=True)
class ExplorationParameters:
    max_epsilon: float
    min_epsilon: float
    epsilon_decay: int


@dataclass(frozen=True)
class ReplayMemoryParameters:
    alpha: float
    max_beta: float
    min_beta: float
    beta_decay: int


@dataclass(frozen=True)
class AgentArchitecture:
    embedding: NetworkArchitecture
    advantage_stream: NetworkArchitecture
    value_stream: NetworkArchitecture
    exploration: ExplorationParameters
    replay_memory: ReplayMemoryParameters
    learning_rate: float
    learning_frequency: int
    target_network_weight_replace_steps: int
    gamma: float
    huber_loss_delta: float
    batch_size: int

    def __init__(
        self,
        agent_dict: EventuallyNestedDict = None,
        network_architecture_implementation_folder_path: Optional[str] = None,
        network_architecture_frame_folder_path: Optional[str] = None,
        load_from_dict: dict = None,
    ):
        if load_from_dict is not None:
            object.__setattr__(
                self,
                "embedding",
                NetworkArchitecture(load_from_dict=load_from_dict["embedding"]),
            )
            object.__setattr__(
                self,
                "advantage_stream",
                NetworkArchitecture(load_from_dict=load_from_dict["advantage_stream"]),
            )
            object.__setattr__(
                self,
                "value_stream",
                NetworkArchitecture(load_from_dict=load_from_dict["value_stream"]),
            )
            agent_dict = load_from_dict

        else:
            object.__setattr__(
                self,
                "embedding",
                NetworkArchitecture(
                    network=agent_dict["embedding"],
                    implementation_folder_path=network_architecture_implementation_folder_path,
                    frame_folder_path=network_architecture_frame_folder_path,
                ),
            )
            object.__setattr__(
                self,
                "advantage_stream",
                NetworkArchitecture(
                    network=agent_dict["advantage_stream"],
                    implementation_folder_path=network_architecture_implementation_folder_path,
                    frame_folder_path=network_architecture_frame_folder_path,
                ),
            )
            object.__setattr__(
                self,
                "value_stream",
                NetworkArchitecture(
                    network=agent_dict["value_stream"],
                    implementation_folder_path=network_architecture_implementation_folder_path,
                    frame_folder_path=network_architecture_frame_folder_path,
                ),
            )

        object.__setattr__(
            self, "exploration", ExplorationParameters(**agent_dict["exploration"])
        )
        object.__setattr__(
            self, "replay_memory", ReplayMemoryParameters(**agent_dict["replay_memory"])
        )
        object.__setattr__(self, "learning_rate", agent_dict["learning_rate"])
        object.__setattr__(self, "learning_frequency", agent_dict["learning_frequency"])
        object.__setattr__(
            self,
            "target_network_weight_replace_steps",
            agent_dict["target_network_weight_replace_steps"],
        )
        object.__setattr__(self, "gamma", agent_dict["gamma"])
        object.__setattr__(self, "huber_loss_delta", agent_dict["huber_loss_delta"])
        object.__setattr__(self, "batch_size", agent_dict["batch_size"])
