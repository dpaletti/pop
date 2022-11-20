from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union, Tuple

import toml
import json

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
    expert_only: bool = False
    do_nothing: bool = False

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
        if model_dict.get("expert_only"):
            object.__setattr__(self, "expert_only", model_dict["expert_only"])
        if model_dict.get("do_nothing"):
            object.__setattr__(self, "do_nothing", model_dict["do_nothing"])


@dataclass(frozen=True)
class TrainingParameters:
    steps: int
    train: bool
    tensorboard_dir: str
    curriculum: bool
    reset_decay: bool
    save_frequency: int
    local: bool = False
    pre_train: bool = False
    chronics: Union[int, str] = -1


@dataclass(frozen=True)
class EvaluationParameters:
    episodes: int
    evaluation_dir: str
    generate_grid2viz_data: str
    compute_score: str
    score: str = "2022"


@dataclass(frozen=True)
class LoadingParameters:
    load: bool
    load_dir: str
    reset_exploration: bool = False


@dataclass(frozen=True)
class EnvironmentParameters:
    name: str
    reward: str
    difficulty: Union[int, str]
    feature_ranges: Dict[str, Tuple[float, float]]

    def __init__(
        self,
        environment_dict: Dict[str, Union[str, bool, int, float, Dict[str, float]]],
    ):
        object.__setattr__(self, "name", environment_dict["name"])
        object.__setattr__(self, "difficulty", environment_dict["difficulty"])
        object.__setattr__(
            self,
            "reward",
            environment_dict["reward"],
        )
        object.__setattr__(
            self,
            "feature_ranges",
            {
                node_or_edge: {
                    feature_name: tuple(feature_range)
                    for feature_name, feature_range in features.items()
                }
                for node_or_edge, features in dict(
                    json.loads(Path(environment_dict["feature_ranges"]).read_text())
                ).items()
            },
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
