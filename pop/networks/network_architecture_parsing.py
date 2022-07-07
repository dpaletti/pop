from typing import Union

from configs.network_architecture import NetworkLayer, NetworkArchitecture
import re


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pop.networks.gcn import GCN
    from networks.dueling_net import DuelingNet

# Necessary imports (with aliases) for parsing
import torch.nn as nn
import dgl.nn.pytorch as dgl_nn
import pop.networks.custom_layers as cl  # pylint: disable=unused-import


def get_network(
    self: Union["GCN", "DuelingNet"],
    architecture: NetworkArchitecture,
    is_graph_network: bool,
) -> dgl_nn.Sequential:
    if is_graph_network:
        model: dgl_nn.Sequential = dgl_nn.Sequential()
    else:
        model: nn.Sequential = nn.Sequential()
    for layer in architecture.layers:
        model.append(eval(_parse_layer(self, layer)))
    return model


def _parse_layer(self: Union["GCN", "DuelingNet"], layer: NetworkLayer) -> str:
    return (
        (
            "dgl_nn"
            if layer.module == "dgl"
            else "nn"
            if layer.module == "pytorch"
            else "cl"
        )
        + "."
        + layer.type
        + "("
        + ",".join(
            [
                arg_name
                + "="
                + (
                    (_replace_dynamic_placeholder(self, str(arg_value)))
                    if arg_name != "activation"
                    else ("nn." + str(arg_value) + "()")
                )
                for arg_name, arg_value in layer.kwargs.items()
                if arg_name not in {"type", "module"}
            ]
        )
        + ")"
    )


def _replace_dynamic_placeholder(
    self: Union["GCN", "DuelingNet"], placeholder: str
) -> str:
    if not re.match(r"<\w*>", placeholder):
        return placeholder
    return str(self.__getattribute__(re.sub(r"[<>]", "", placeholder)))