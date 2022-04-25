# Adapted from https://github.com/ast0414/pointer-networks-pytorch/blob/master/model.py
import torch as th
from torch import nn

from pointer_network_utilities import masked_max
from pop.attention import Attention
from pop.encoder import Encoder


class PointerNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_size: int,
        bidirectional: bool = True,
        batch_first: bool = True,
    ):
        super(PointerNetwork, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.num_layers = 1
        self.batch_first = batch_first

        # TODO apply Graph NN here
        self.embedding = nn.Linear(
            in_features=input_dim, out_features=embedding_dim, bias=False
        )
        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=self.num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

        self.decoding_rnn = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        self.attn = Attention(hidden_size=hidden_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    th.nn.init.zeros_(m.bias)

    def forward(
        self, input_seq: th.Tensor, input_lengths: th.Tensor
    ) -> (th.Tensor, th.Tensor, th.Tensor):
        if self.batch_first:
            batch_size = input_seq.size(0)
            max_seq_len = input_seq.size(1)
        else:
            batch_size = input_seq.size(1)
            max_seq_len = input_seq.size(0)

        embedded = self.embedding(input_seq)
        encoder_outputs, encoder_hidden = self.encoder(embedded, input_lengths)

        if self.bidirectional:
            encoder_outputs = (
                encoder_outputs[:, :, : self.hidden_size]
                + encoder_outputs[:, :, self.hidden_size :]
            )

        encoder_h_n, encoder_c_n = encoder_hidden
        encoder_h_n = encoder_h_n.view(
            self.num_layers, self.num_directions, batch_size, self.hidden_size
        )
        encoder_c_n = encoder_c_n.view(
            self.num_layers, self.num_directions, batch_size, self.hidden_size
        )

        decoder_input = encoder_outputs.new_zeros(
            th.Size((batch_size, self.hidden_size))
        )
        decoder_hidden = (
            encoder_h_n[-1, 0, :, :].squeeze(),
            encoder_c_n[-1, 0, :, :].squeeze(),
        )

        range_tensor = th.arange(
            max_seq_len, device=input_lengths.device, dtype=input_lengths.dtype
        ).expand(batch_size, max_seq_len, max_seq_len)
        each_len_tensor = input_lengths.view(-1, 1, 1).expand(
            batch_size, max_seq_len, max_seq_len
        )

        row_mask_tensor = th.lt(
            range_tensor, each_len_tensor
        )  # original implementation uses '<' (element-wise less-than)
        col_mask_tensor = row_mask_tensor.transpose(1, 2)
        mask_tensor = row_mask_tensor * col_mask_tensor

        pointer_log_scores = []
        pointer_argmaxs = []

        for i in range(max_seq_len):
            sub_mask = mask_tensor[:, i, :].float()

            h_i, c_i = self.decoding_rnn(decoder_input, decoder_hidden)

            decoder_hidden = (h_i, c_i)

            log_pointer_score = self.attn(h_i, encoder_outputs, sub_mask)
            pointer_log_scores.append(log_pointer_score)

            _, masked_argmax = masked_max(
                log_pointer_score, sub_mask, dim=1, keepdim=True
            )

            pointer_argmaxs.append(masked_argmax)
            index_tensor = masked_argmax.unsqueeze(-1).expand(
                batch_size, 1, self.hidden_size
            )

            decoder_input = th.gather(
                encoder_outputs, dim=1, index=index_tensor
            ).squeeze(1)

        pointer_log_scores = th.stack(pointer_log_scores, 1)
        pointer_argmaxs = th.cat(pointer_argmaxs, 1)

        return pointer_log_scores, pointer_argmaxs, mask_tensor
