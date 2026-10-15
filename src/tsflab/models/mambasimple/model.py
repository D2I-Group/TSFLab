"""Clean-room Mamba forecasting wrapper built from the published block equations."""

from __future__ import annotations

import math

from torch import nn

from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._components.mamba import MambaResidualBlock, RMSNorm
from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Time-token Mamba backbone followed by an explicit horizon projection."""

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        c_out=None,
        features="M",
        d_model=128,
        d_state=16,
        e_layers=2,
        expand=2,
        d_conv=4,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        inner, rank = d_model * expand, math.ceil(d_model / 16)
        self.input_projection = nn.Linear(enc_in, d_model)
        self.input_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [MambaResidualBlock(d_model, inner, rank, d_conv, d_state) for _ in range(e_layers)]
        )
        self.final_norm = RMSNorm(d_model)
        self.channel_projection = nn.Linear(d_model, enc_in)
        self.horizon_projection = nn.Linear(seq_len, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized, mean, scale = standardize(x_enc, detach_std=True)
        hidden = self.input_dropout(self.input_projection(normalized))
        for layer in self.layers:
            hidden = layer(hidden)
        history = self.channel_projection(self.final_norm(hidden))
        forecast = self.horizon_projection(history.transpose(1, 2)).transpose(1, 2)
        return destandardize(forecast, mean, scale)
