"""Independent differentiable elastic-net lag-regression baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """Direct channel-wise forecast with the elastic-net weight penalty."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, penalty: float = 1e-4, l1_ratio: float = 0.5) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("dimensions must be positive and penalty non-negative")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.penalty, self.l1_ratio = penalty, l1_ratio
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
            raise ValueError(f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}")
        forecast = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        weights = self.projection.weight
        regularizer = self.l1_ratio * weights.abs().sum()
        regularizer = regularizer + 0.5 * (1.0 - self.l1_ratio) * weights.square().sum()
        self.aux_loss = self.penalty * regularizer
        return forecast
