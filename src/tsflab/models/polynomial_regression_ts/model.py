"""Independent polynomial lag-regression baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Expand each channel's lags to integer powers, then regress the horizon."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, degree: int = 2) -> None:
        super().__init__()
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
        check_window(x_enc, self.seq_len, self.enc_in)
        forecast = self.projection(self.polynomial_features(x_enc)).transpose(1, 2)
        self.aux_loss = forecast.new_zeros(())
        return forecast
