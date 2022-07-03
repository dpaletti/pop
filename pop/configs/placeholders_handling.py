import re
from typing import Dict, Union, List
import numexpr as ne


def replace_backward_reference(
    reference_dict: Dict[str, Dict[str, Union[int, bool, float, str]]],
    value_to_replace: Union[int, float, bool, str],
    evaluate_expressions: bool,
) -> Union[int, float, bool, str]:
    if not isinstance(value_to_replace, str):
        return value_to_replace

    # Backward references are "<<...>>" where ...="layerName_paramName"
    # A general param_value may contain multiple backward references arranged
    # as a mathematical expression
    backward_references: List[str] = re.findall(r"<<\w*>>", value_to_replace)

    if not backward_references:
        return value_to_replace

    for backward_reference in backward_references:

        # strip "<" and ">" from the references
        backward_reference_stripped: str = re.sub(r"[<>]", "", backward_reference)

        # split over "_"
        # First part is layer name
        # All the rest is layer parameter name
        split_backward_reference: List[str] = backward_reference_stripped.split("_")

        value_to_replace = value_to_replace.replace(
            backward_reference,
            str(
                reference_dict[split_backward_reference[0]][
                    "_".join(split_backward_reference[1:])
                ]
            ),
        )
    if evaluate_expressions:
        return ne.evaluate(value_to_replace)
    else:
        return value_to_replace


def replace_placeholders(
    implementation_dict: Dict[
        str,
        Union[str, Dict[str, Union[int, float, str, bool]]],
    ],
    frame_dict: Dict[str, Dict[str, str]],
) -> Dict[str, Dict[str, Union[int, bool, float, str]]]:
    return {
        layer_name: {
            layer_param_name: (
                layer_param_value
                if not layer_param_value == "..."
                else implementation_dict[layer_name][layer_param_name]
            )
            for layer_param_name, layer_param_value in layer_param_dict.items()
        }
        for layer_name, layer_param_dict in frame_dict.items()
    }
