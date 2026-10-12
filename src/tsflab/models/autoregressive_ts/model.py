"""Independent direct multi-horizon autoregressive baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Apply one shared lag regression to every channel independently."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.projection = nn.Linear(seq_len, pred_len)
        self.aux_loss: torch.Tensor | None = None

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        forecast = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        self.aux_loss = forecast.new_zeros(())
        return forecast
