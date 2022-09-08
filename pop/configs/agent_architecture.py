import abc
from dataclasses import dataclass
from functools import reduce
from typing import Optional, ClassVar, List, Type, Any

from pop.configs.network_architecture import NetworkArchitecture
from pop.configs.type_aliases import EventuallyNestedDict


@dataclass(frozen=True)
class ExplorationParameters(abc.ABC):
    method: ClassVar[str]

    def __init__(self, d: dict):
        ...

    @staticmethod
    @abc.abstractmethod
    def network_architecture_fields() -> List[List[str]]:
        ...


@dataclass(frozen=True)
class InverseModelArchitecture:
    embedding: NetworkArchitecture
    action_prediction_stream: NetworkArchitecture
    learning_rate: int


@dataclass(frozen=True)
class RandomNetworkDistillerArchitecture:
    network: NetworkArchitecture
    learning_rate: int


@dataclass(frozen=True)
class EpisodicMemoryParameters(ExplorationParameters):
    method: ClassVar[str] = "episodic_memory"
    size: int
    neighbors: int
    exploration_bonus_limit: int
    random_network_distiller: RandomNetworkDistillerArchitecture
    inverse_model: InverseModelArchitecture

    @staticmethod
    def network_architecture_fields() -> List[List[str]]:
        return [
            ["random_network_distiller", "network"],
            ["inverse_model", "embedding"],
            ["inverse_model", "action_prediction_stream"],
        ]

    def __init__(self, d: dict):
        super(EpisodicMemoryParameters, self).__init__(d)
        object.__setattr__(self, "size", d["size"])
        object.__setattr__(self, "neighbors", d["neighbors"])
        object.__setattr__(
            self, "exploration_bonus_limit", d["exploration_bonus_limit"]
        )
        object.__setattr__(
            self, "inverse_model", InverseModelArchitecture(**d["inverse_model"])
        )
        object.__setattr__(
            self,
            "random_network_distiller",
            RandomNetworkDistillerArchitecture(**d["random_network_distiller"]),
        )


@dataclass(frozen=True)
class EpsilonGreedyParameters(ExplorationParameters):
    method: ClassVar[str] = "epsilon_greedy"
    max_epsilon: float
    min_epsilon: float
    epsilon_decay: float

    def __init__(self, d: dict):
        super(EpsilonGreedyParameters, self).__init__(d)
        object.__setattr__(self, "max_epsilon", d["max_epsilon"])
        object.__setattr__(self, "min_epsilon", d["min_epsilon"])
        object.__setattr__(self, "epsilon_decay", d["epsilon_decay"])

    @staticmethod
    def network_architecture_fields() -> List[List[str]]:
        return []


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
        exploration_module_cls: Type[ExplorationParameters] = next(
            filter(
                lambda subclass: subclass.method == exploration_method,
                available_exploration_methods,
            )
        )

        for (
            network_architecture_keys
        ) in exploration_module_cls.network_architecture_fields():
            network_architecture = self._deep_get(
                agent_dict["exploration"], network_architecture_keys
            )
            parsed_network_architecture = NetworkArchitecture(
                network=network_architecture,
                implementation_folder_path=network_architecture_implementation_folder_path,
                frame_folder_path=network_architecture_frame_folder_path,
            )
            self._deep_update(
                agent_dict["exploration"],
                network_architecture_keys,
                parsed_network_architecture,
            )
        object.__setattr__(
            self,
            "exploration",
            exploration_module_cls(
                {
                    parameter: parameter_value
                    for parameter, parameter_value in agent_dict["exploration"].items()
                    if parameter != "method"
                },
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

    @staticmethod
    def _deep_get(di: dict, keys: List[str]):
        return reduce(lambda d, key: d.get(key) if d else None, keys, di)

    @staticmethod
    def _deep_update(mapping: dict, keys: List[str], value: Any) -> dict:
        k = keys[0]
        if k in mapping and isinstance(mapping[k], dict) and len(keys) > 1:
            mapping[k] = AgentArchitecture._deep_update(mapping[k], keys[1:], value)
        else:
            mapping[k] = value
            return mapping
