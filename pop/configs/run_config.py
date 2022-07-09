from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import toml

from configs.architecture import Architecture
from configs.placeholders_handling import replace_backward_reference

ParsedTOMLDict = Dict[str, Dict[str, Union[int, float, bool, str]]]


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


@dataclass(frozen=True)
class EvaluationParameters:
    episodes: int
    evaluation_dir: str


@dataclass(frozen=True)
class LoadingParameters:
    load: bool
    load_dir: str


@dataclass(frozen=True)
class RewardSpecification:
    reward_components: Dict[str, float]


@dataclass(frozen=True)
class EnvironmentParameters:
    name: str
    reward: RewardSpecification

    def __init__(
        self,
        environment_dict: Dict[str, Union[str, bool, int, float, Dict[str, float]]],
    ):
        object.__setattr__(self, "name", environment_dict["name"])
        object.__setattr__(
            self,
            "reward",
            RewardSpecification(reward_components=environment_dict["reward"]),
        )


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

        run_config_full_dict = {
            section_name: {
                param_name: replace_backward_reference(
                    run_config_dict, param_value, evaluate_expressions=False
                )
                for param_name, param_value in section_param_dict.items()
            }
            for section_name, section_param_dict in run_config_dict.items()
        }

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
