"""Independent direct multi-horizon autoregressive baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """Apply one shared lag regression to every channel independently."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int) -> None:
        super().__init__()
        if seq_len < 1 or pred_len < 1:
            raise ValueError("seq_len and pred_len must be positive")
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
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )
        forecast = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        self.aux_loss = forecast.new_zeros(())
        return forecast
