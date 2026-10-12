"""Paper-driven local implementation of the LTSF-Linear baseline."""

from __future__ import annotations

from torch import nn

from tsflab.models._components.channel_wise_linear import ChannelWiseLinear


class Model(nn.Module):
    """Apply one history-to-horizon affine map independently to each channel."""

    def __init__(self, c_in: int, seq_len: int, pred_len: int, individual: bool = False):
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("channels and sequence lengths must be positive")
        self.c_in = c_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.projection = ChannelWiseLinear(seq_len, pred_len, c_in, individual)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.shape[1:] != (self.seq_len, self.c_in):
            raise ValueError("x_enc does not match configured time/channel dimensions")
        return self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
