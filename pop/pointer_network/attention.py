# Adapted from https://github.com/ast0414/pointer-networks-pytorch/blob/master/model.py

import torch as th
from torch import nn

from pointer_network_utilities import masked_log_softmax


class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.w1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(
        self, decoder_state: th.Tensor, encoder_outputs: th.Tensor, mask: th.Tensor
    ) -> th.Tensor:
        encoder_transform = self.W1(encoder_outputs)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)

        u_i = self.vt(th.tanh(encoder_transform + decoder_transform)).unsqueeze(1)

        log_score = masked_log_softmax(u_i, mask, dim=-1)

        return log_score
