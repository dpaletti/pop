from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union, Optional

import toml

from pop.configs.placeholders_handling import (
    replace_backward_reference,
    replace_placeholders,
)
from pop.configs.type_aliases import ParsedTOMLDict


@dataclass(frozen=True)
class NetworkLayer:
    name: str
    type: str
    module: str
    kwargs: Dict[str, Union[int, float, bool, str]]


@dataclass(frozen=True)
class NetworkArchitecture:
    layers: List[NetworkLayer]

    def __init__(
        self,
        load_from_dict: Optional[Dict[str, List[ParsedTOMLDict]]] = None,
        network: Optional[str] = None,
        implementation_folder_path: Optional[str] = None,
        frame_folder_path: Optional[str] = None,
    ):
        if load_from_dict:
            object.__setattr__(
                self,
                "layers",
                [NetworkLayer(**layer) for layer in load_from_dict["layers"]],
            )
            return

        if (
            network is None
            or implementation_folder_path is None
            or frame_folder_path is None
        ):
            raise Exception("Pleas pass either layers or all the other parameters")

        # Loading implementation value with actual architecture values
        network_architecture_implementation_dict: Dict[
            str, Union[str, Dict[str, Union[int, float, str, bool]]]
        ] = toml.load(open(Path(implementation_folder_path, network + ".toml")))

        assert "frame" in network_architecture_implementation_dict.keys()

        # Loading architecture frame with placeholders and back references
        # Placeholders are replaced with implementation values
        network_architecture_frame_dict: Dict[str, Dict[str, str]] = toml.load(
            open(
                Path(
                    frame_folder_path,
                    network_architecture_implementation_dict["frame"] + ".toml",
                )
            )
        )

        assert set(
            {k for k in network_architecture_implementation_dict.keys() if k != "frame"}
        ).issubset(network_architecture_frame_dict.keys())

        no_placeholder_architecture: Dict[
            str, Dict[str, int, bool, float]
        ] = replace_placeholders(
            implementation_dict=network_architecture_implementation_dict,
            frame_dict=network_architecture_frame_dict,
        )

        full_architecture = {
            layer_name: {
                layer_param_name: replace_backward_reference(
                    no_placeholder_architecture,
                    layer_param_value,
                    evaluate_expressions=True,
                )
                for layer_param_name, layer_param_value in layer_param_dict.items()
            }
            for layer_name, layer_param_dict in no_placeholder_architecture.items()
        }

        object.__setattr__(
            self,
            "layers",
            [
                NetworkLayer(
                    **{
                        "name": layer_name,
                        "type": layer_params["type"],
                        "module": layer_params["module"],
                        "kwargs": {
                            layer_param_name: layer_param_value
                            for layer_param_name, layer_param_value in layer_params.items()
                            if layer_param_name not in {"layer", "module"}
                        },
                    }
                )
                for layer_name, layer_params in full_architecture.items()
            ],
        )
