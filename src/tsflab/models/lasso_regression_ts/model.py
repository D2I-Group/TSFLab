"""Independent Lasso-regularized lag regression baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """Channel-independent linear forecast with an L1 training penalty."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, l1_penalty: float = 1e-5) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("dimensions must be positive")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.l1_penalty = l1_penalty
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
        self.aux_loss = self.l1_penalty * self.projection.weight.abs().sum()
        return forecast
