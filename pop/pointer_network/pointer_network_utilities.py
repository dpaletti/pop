# adapted from https://github.com/ast0414/pointer-networks-pytorch/blob/master/model.py
import torch as th
from torch.nn import functional


def masked_log_softmax(vector: th.Tensor, mask: th.Tensor, dim: int = -1) -> th.Tensor:
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim:
            mask = mask.unsqueeze(1)
        vector = vector + (mask + 1e-45).log()  # zero out masked elements in logspace
    return functional.log_softmax(vector, dim=dim)


def masked_max(
    vector: th.Tensor,
    mask: th.Tensor,
    dim: int,
    keepdim: bool = False,
    min_val: float = -1e7,
) -> (th.Tensor, th.Tensor):
    one_minus_mask = (1.0 - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    max_value, max_index = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value, max_index
