"""Clean-room RLinear implementation from the paper's affine baseline."""

from __future__ import annotations

import torch.nn as nn

from tsflab.models._components.channel_wise_linear import ChannelWiseLinear
from tsflab.models._components.revin import RevIN


class Model(nn.Module):
    """RevIN followed by the paper's channel-independent affine map."""

    def __init__(
        self,
        c_in: int,
        seq_len: int,
        pred_len: int,
        individual: bool = False,
        affine: bool = False,
        subtract_last: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) <= 0:
            raise ValueError("sequence lengths must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = c_in
        self.normalization = RevIN(
            c_in, affine=affine, subtract_last=subtract_last
        )
        self.input_dropout = nn.Dropout(dropout)
        self.projection = ChannelWiseLinear(seq_len, pred_len, c_in, individual)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("RLinear expects (batch, configured seq_len, c_in)")
        normalized = self.normalization(x_enc, "norm")
        forecast = self.projection(
            self.input_dropout(normalized).transpose(1, 2)
        ).transpose(1, 2)
        return self.normalization(forecast, "denorm")
