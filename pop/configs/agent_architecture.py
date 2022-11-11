import abc
from dataclasses import dataclass
from functools import reduce
from typing import Any, ClassVar, List, Optional, Type

from pop.configs.network_architecture import NetworkArchitecture
from pop.configs.type_aliases import EventuallyNestedDict


@dataclass(frozen=True)
class ExplorationParameters(abc.ABC):
    @abc.abstractmethod
    def __init__(self, d: dict):
        ...

    @staticmethod
    @abc.abstractmethod
    def network_architecture_fields() -> List[List[str]]:
        ...

    @staticmethod
    @abc.abstractmethod
    def get_method() -> str:
        ...


@dataclass(frozen=True)
class InverseModelArchitecture:
    embedding: NetworkArchitecture
    action_prediction_stream: NetworkArchitecture
    learning_rate: float
    adam_epsilon: float


@dataclass(frozen=True)
class RandomNetworkDistillerArchitecture:
    network: NetworkArchitecture
    learning_rate: float
    adam_epsilon: float


@dataclass(frozen=True)
class EpisodicMemoryParameters(ExplorationParameters):
    method: str
    size: int
    neighbors: int
    exploration_bonus_limit: int
    maximum_similarity: float
    random_network_distiller: RandomNetworkDistillerArchitecture
    inverse_model: InverseModelArchitecture

    @staticmethod
    def get_method() -> str:
        return "episodic_memory"

    @staticmethod
    def network_architecture_fields() -> List[List[str]]:
        return [
            ["random_network_distiller", "network"],
            ["inverse_model", "embedding"],
            ["inverse_model", "action_prediction_stream"],
        ]

    def __init__(self, d: dict):
        super().__init__(d)
        object.__setattr__(self, "method", d["method"])
        object.__setattr__(self, "size", d["size"])
        object.__setattr__(self, "neighbors", d["neighbors"])
        object.__setattr__(self, "maximum_similarity", d["maximum_similarity"])
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
    method: str
    max_epsilon: float
    min_epsilon: float
    epsilon_decay: float

    @staticmethod
    def get_method() -> str:
        return "epsilon_greedy"

    def __init__(self, d: dict):
        super().__init__(d)
        object.__setattr__(self, "method", d["method"])
        object.__setattr__(self, "max_epsilon", d["max_epsilon"])
        object.__setattr__(self, "min_epsilon", d["min_epsilon"])
        object.__setattr__(self, "epsilon_decay", d["epsilon_decay"])

    @staticmethod
    def network_architecture_fields() -> List[List[str]]:
        return []


@dataclass(frozen=True)
class EpsilonEpisodicParameters(
    EpisodicMemoryParameters, EpsilonGreedyParameters, ExplorationParameters
):
    @staticmethod
    def get_method() -> str:
        return "epsilon_episodic"

    def __init__(self, d: dict):
        super().__init__(d)

    @staticmethod
    def network_architecture_fields() -> List[List[str]]:
        return EpisodicMemoryParameters.network_architecture_fields()


@dataclass(frozen=True)
class ReplayMemoryParameters:
    alpha: float
    max_beta: float
    min_beta: float
    annihilation_rate: int
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
    adam_epsilon: float
    target_network_weight_replace_steps: int
    gamma: float
    huber_loss_delta: float
    batch_size: int
    intrinsic_reward_relevance: float = 0

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

            exploration_module_cls = AgentArchitecture._get_exploration_module_cls(
                load_from_dict["exploration"]
            )
            AgentArchitecture._parse_network_architectures(
                d=load_from_dict["exploration"],
                exploration_module_cls=exploration_module_cls,
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

            exploration_module_cls = AgentArchitecture._get_exploration_module_cls(
                agent_dict["exploration"]
            )

            AgentArchitecture._parse_network_architectures(
                d=agent_dict["exploration"],
                exploration_module_cls=exploration_module_cls,
                implementation_folder_path=network_architecture_implementation_folder_path,
                frame_folder_path=network_architecture_frame_folder_path,
            )

        object.__setattr__(
            self,
            "exploration",
            exploration_module_cls(agent_dict["exploration"]),
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
        object.__setattr__(self, "adam_epsilon", agent_dict["adam_epsilon"])
        object.__setattr__(self, "huber_loss_delta", agent_dict["huber_loss_delta"])
        object.__setattr__(self, "batch_size", agent_dict["batch_size"])
        if agent_dict.get("intrinsic_reward_relevance"):
            object.__setattr__(
                self,
                "intrinsic_reward_relevance",
                agent_dict["intrinsic_reward_relevance"],
            )

    @staticmethod
    def _parse_network_architectures(
        d: dict,
        exploration_module_cls: Type[ExplorationParameters],
        implementation_folder_path: Optional[str] = None,
        frame_folder_path: Optional[str] = None,
    ) -> None:

        for (
            network_architecture_keys
        ) in exploration_module_cls.network_architecture_fields():
            network_architecture = AgentArchitecture._deep_get(
                d, network_architecture_keys
            )
            parsed_network_architecture = (
                NetworkArchitecture(
                    network=network_architecture,
                    implementation_folder_path=implementation_folder_path,
                    frame_folder_path=frame_folder_path,
                )
                if implementation_folder_path and frame_folder_path
                else NetworkArchitecture(load_from_dict=network_architecture)
            )
            AgentArchitecture._deep_update(
                d,
                network_architecture_keys,
                parsed_network_architecture,
            )

    @staticmethod
    def _get_exploration_module_cls(d: dict) -> Type[ExplorationParameters]:
        exploration_method = d.get("method")
        available_exploration_methods = [
            subclass for subclass in ExplorationParameters.__subclasses__()
        ]
        if exploration_method is None:
            raise Exception(
                "Missing Exploration method."
                + "\nAvailable methods are: "
                + str(available_exploration_methods)
            )
        return next(
            filter(
                lambda subclass: subclass.get_method() == exploration_method,
                available_exploration_methods,
            )
        )

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
