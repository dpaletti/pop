import abc
from dataclasses import dataclass
from typing import Optional, ClassVar

from pop.configs.network_architecture import NetworkArchitecture
from pop.configs.type_aliases import EventuallyNestedDict


@dataclass(frozen=True)
class ExplorationParameters(abc.ABC):
    method: ClassVar[str]


@dataclass(frozen=True)
class InverseModelArchitecture:
    embedding: NetworkArchitecture
    action_prediction_stream: NetworkArchitecture


@dataclass(frozen=True)
class EpisodicMemoryArchitecture(ExplorationParameters):
    method: ClassVar[str] = "episodic memory"
    size: int
    neighbors: int
    exploration_bonus_limit: int
    inverse_model: InverseModelArchitecture
    random_network_distiller: NetworkArchitecture


@dataclass(frozen=True)
class EpsilonGreedyParameters(ExplorationParameters):
    method: ClassVar[str] = "epsilon greedy"

    max_epsilon: float
    min_epsilon: float
    epsilon_decay: float

    def __post_init__(self):
        if self.max_epsilon < 0 or self.max_epsilon > 1:
            raise Exception(
                "Invalid value encountered for max_epsilon: "
                + str(self.max_epsilon)
                + "\n max_epsilon must be in [0, 1]"
            )
        if self.min_epsilon < 0 or self.min_epsilon > 1:
            raise Exception(
                "Invalid value encountered for min_epsilon: "
                + str(self.max_epsilon)
                + "\n min_epsilon must be in [0, 1]"
            )
        if self.epsilon_decay < 0:
            raise Exception(
                "Invalid value encountered for epsilon_decay: "
                + str(self.epsilon_decay)
                + "\n epsilon_decay must be greater than 0"
            )


@dataclass(frozen=True)
class ReplayMemoryParameters:
    alpha: float
    max_beta: float
    min_beta: float
    beta_decay: int
    capacity: int


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

        exploration_method = agent_dict["exploration"].get("method")
        available_exploration_methods = [
            subclass for subclass in ExplorationParameters.__subclasses__()
        ]
        if exploration_method is None:
            raise Exception(
                "Invalid method in exploration_section (may be missing): "
                + str([subclass.method for subclass in available_exploration_methods])
                + "\nAvailable methods are: "
                + str(available_exploration_methods)
            )
        object.__setattr__(
            self,
            "exploration",
            next(
                filter(
                    lambda subclass: subclass.method == exploration_method,
                    available_exploration_methods,
                )
            )(
                **{
                    parameter: parameter_value
                    for parameter, parameter_value in agent_dict["exploration"].items()
                    if parameter != "method"
                }
            ),
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
