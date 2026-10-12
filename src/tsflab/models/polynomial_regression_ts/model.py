"""Independent polynomial lag-regression baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """Expand each channel's lags to integer powers, then regress the horizon."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, degree: int = 2) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("dimensions and degree must be positive")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.degree = degree
        self.projection = nn.Linear(seq_len * degree, pred_len)
        self.aux_loss: torch.Tensor | None = None

    def polynomial_features(self, x: torch.Tensor) -> torch.Tensor:
        channel_first = x.transpose(1, 2)
        return torch.cat(
            [channel_first.pow(power) for power in range(1, self.degree + 1)], dim=-1
        )

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
        forecast = self.projection(self.polynomial_features(x_enc)).transpose(1, 2)
        self.aux_loss = forecast.new_zeros(())
        return forecast
