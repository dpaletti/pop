# Adapted from https://github.com/ast0414/pointer-networks-pytorch/blob/master/model.py
import torch as th
from torch import nn


class Encoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int = 1,
        batch_first: bool = True,
        bidirectional: bool = True,
    ):
        super(Encoder, self).__init__()

        self.batch_first = batch_first
        self.rnn = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=batch_first,
            bidirectional=bidirectional,
        )

    def forward(
        self, embedded_inputs: th.Tensor, input_lengths: th.Tensor
    ) -> (th.Tensor, th.Tensor):

        rnn_input = nn.utils.rnn.pack_padded_sequence(
            embedded_inputs, input_lengths, batch_first=self.batch_first
        )
        outputs, hidden = self.rnn(rnn_input)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=self.batch_first
        )
        return outputs, hidden
