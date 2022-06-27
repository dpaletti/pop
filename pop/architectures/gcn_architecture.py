from typing import Dict, Union, List
import toml
from toml import TomlDecodeError
from dataclasses import dataclass, InitVar, field
from typeguard import typechecked


@typechecked
@dataclass(frozen=True)
class POPArchitecture:
    agent_neighbourhood_radius: int
    decentralized: bool
    fixed_communities: bool
    epsilon_beta_scheduling: bool
    agent_type: str


@typechecked
@dataclass(frozen=True)
class NetworkLayer:
    name: str
    layer: str
    module: str
    kwargs: Dict[str, Union[int, float, str, bool]]


# TODO: here needs to be finished
@typechecked
@dataclass(frozen=True)
class NetworkArchitecture:
    layers: List[NetworkLayer]

    def __init__(self, network_path: str):
        network_architecture_dict: Dict[
            str, Union[int, float, str, bool]
        ] = load_toml_architecture(network_path)
        _layers = []
        for key, value in network_architecture_dict.items():
            _layers.append(
                NetworkLayer(
                    {"name": key, "module": network_architecture_dict["module"]}
                )
            )


@typechecked
@dataclass(frozen=True)
class ExplorationParameters:
    max_epsilon: float
    min_epsilon: float
    epsilon_decay: int


@typechecked
@dataclass(frozen=True)
class ReplayMemoryParameters:
    alpha: float
    max_beta: float
    min_beta: float
    beta_decay: int


@typechecked
@dataclass(frozen=True)
class AgentArchitecture:
    network: NetworkArchitecture = field(init=False)
    network_path: InitVar[str]
    exploration_parameters: ExplorationParameters = field(init=False)
    exploration: InitVar[Dict[str, Union[int, float]]]
    replay_memory_parameters: ReplayMemoryParameters = field(init=False)
    replay_memory: InitVar[Dict[str, Union[int, float]]]
    learning_rate: float
    learning_frequency: int
    target_network_weight_replace_steps: int
    gamma: float
    huber_loss_delta: float
    batch_size: int

    def __post_init__(
        self,
        network_path: str,
        exploration: Dict[str, Union[int, float]],
        replay_memory: Dict[str, Union[int, float]],
    ):
        object.__setattr__(self, "network", NetworkArchitecture(network_path))
        object.__setattr__(
            self, "exploration_parameters", ExplorationParameters(**exploration)
        )
        object.__setattr__(
            self, "replay_memory_parameters", ReplayMemoryParameters(**replay_memory)
        )


@typechecked
@dataclass(frozen=True)
class Architecture:
    pop_architecture: POPArchitecture = field(init=False)
    pop: InitVar[Dict[str, Union[int, float, str, bool]]]

    agent_architecture: AgentArchitecture = field(init=False)
    agent: InitVar[Dict[str, Union[int, float, str, bool]]]

    manager_architecture: AgentArchitecture = field(init=False)
    manager: InitVar[Dict[str, Union[int, float, str, bool]]]

    head_manager_architecture: AgentArchitecture = field(init=False)
    head_manager: InitVar[Dict[str, Union[int, float, str, bool]]]

    def __post_init__(self, pop, agent, manager, head_manager):
        object.__setattr__(self, "pop_architecture", POPArchitecture(**pop))
        object.__setattr__(self, "agent_architecture", AgentArchitecture(**agent))
        object.__setattr__(self, "manager_architecture", AgentArchitecture(**manager))
        object.__setattr__(
            self, "head_manager_architecture", AgentArchitecture(**head_manager)
        )


@typechecked
def parse_architecture(path: str) -> Architecture:
    architecture_dict: Dict[str, Union[int, float, bool, str]] = load_toml_architecture(
        path
    )
    return Architecture(architecture_dict)


@typechecked
def load_toml_architecture(path: str) -> Dict[str, Union[int, float, bool, str]]:
    try:
        architecture_dict = toml.load(open(path))
        return architecture_dict

    except FileNotFoundError:
        print("Could not find model architecture file: " + str(path))
        print("Input a valid architecture file.")
    except TomlDecodeError:
        print("Could not decode: " + str(path))
        print("Input a valid toml file as architecture specification.")
