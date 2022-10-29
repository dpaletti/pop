from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import toml

from pop.configs.architecture import Architecture
from pop.configs.placeholders_handling import replace_backward_reference
from pop.configs.type_aliases import ParsedTOMLDict


@dataclass(frozen=True)
class Reproducibility:
    seed: int
    device: str


@dataclass(frozen=True)
class ModelParameters:
    name: str
    architecture: Architecture
    data_dir: str
    checkpoint_dir: str

    def __init__(self, model_dict: Dict[str, Union[int, bool, str, float]]):
        architecture_path = model_dict["architecture_path"]
        architecture_frame_path: str = str(
            Path(Path(architecture_path).parents[0], "frames")
        )
        architecture_implementation_path: str = str(
            Path(Path(architecture_path).parents[0], "implementations")
        )
        object.__setattr__(self, "name", model_dict["name"])
        object.__setattr__(
            self,
            "architecture",
            Architecture(
                path=architecture_path,
                network_architecture_implementation_folder_path=architecture_implementation_path,
                network_architecture_frame_folder_path=architecture_frame_path,
            ),
        )
        object.__setattr__(self, "data_dir", model_dict["data_dir"])
        object.__setattr__(self, "checkpoint_dir", model_dict["checkpoint_dir"])


@dataclass(frozen=True)
class TrainingParameters:
    steps: int
    train: bool
    tensorboard_dir: str
    curriculum: bool
    reset_decay: bool
    save_frequency: int
    skip: int = 1
    local: bool = False
    pre_train: bool = False
    chronics: Union[int, str] = -1


@dataclass(frozen=True)
class EvaluationParameters:
    episodes: int
    evaluation_dir: str
    generate_grid2viz_data: str
    compute_score: str


@dataclass(frozen=True)
class LoadingParameters:
    load: bool
    load_dir: str
    reset_exploration: bool = False


@dataclass(frozen=True)
class RewardSpecification:
    reward_components: Dict[str, float]


@dataclass(frozen=True)
class EnvironmentParameters:
    name: str
    reward: RewardSpecification
    difficulty: Union[int, str]

    def __init__(
        self,
        environment_dict: Dict[str, Union[str, bool, int, float, Dict[str, float]]],
    ):
        object.__setattr__(self, "name", environment_dict["name"])
        object.__setattr__(self, "difficulty", environment_dict["difficulty"])
        object.__setattr__(
            self,
            "reward",
            RewardSpecification(reward_components=environment_dict["reward"]),
        )


def replace_all_backward_references(run_config_dict: ParsedTOMLDict):
    run_config_full_dict = {}
    for section_name, section_param_dict in run_config_dict.items():
        run_config_full_dict[section_name] = {}
        for param_name, param_value in section_param_dict.items():
            run_config_full_dict[section_name][param_name] = replace_backward_reference(
                run_config_full_dict, param_value, evaluate_expressions=False
            )
    return run_config_full_dict


@dataclass(frozen=True)
class RunConfiguration:
    reproducibility: Reproducibility
    model: ModelParameters
    training: TrainingParameters
    evaluation: EvaluationParameters
    loading: LoadingParameters
    environment: EnvironmentParameters

    def __init__(self, path: str):
        run_config_dict: ParsedTOMLDict = toml.load(open(path))
        run_config_full_dict = replace_all_backward_references(run_config_dict)

        object.__setattr__(
            self,
            "reproducibility",
            Reproducibility(**run_config_full_dict["reproducibility"]),
        )
        object.__setattr__(
            self, "model", ModelParameters(run_config_full_dict["model"])
        )
        object.__setattr__(
            self, "training", TrainingParameters(**run_config_full_dict["training"])
        )
        object.__setattr__(
            self,
            "evaluation",
            EvaluationParameters(**run_config_full_dict["evaluation"]),
        )
        object.__setattr__(
            self, "loading", LoadingParameters(**run_config_full_dict["loading"])
        )
        object.__setattr__(
            self,
            "environment",
            EnvironmentParameters(run_config_full_dict["environment"]),
        )
