"""Paper-driven local implementation of cross-period sparse forecasting."""

from __future__ import annotations

import math

from torch import nn

from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Aggregate locally, forecast phase-aligned subsequences, and interleave them."""

    def __init__(
        self, seq_len: int, pred_len: int, enc_in: int, period: int = 24, d_model: int = 64, model_type: str = "linear"
    ) -> None:
        super().__init__()
        if period > seq_len:
            raise ValueError("period must not exceed seq_len")
        self.seq_len, self.pred_len, self.enc_in, self.period = seq_len, pred_len, enc_in, period
        self.history_periods = seq_len // period
        self.future_periods = math.ceil(pred_len / period)
        kernel = 2 * (period // 2) + 1
        self.aggregation = nn.Conv1d(1, 1, kernel, padding=kernel // 2)
        self.forecaster = (
            nn.Linear(self.history_periods, self.future_periods)
            if model_type == "linear"
            else nn.Sequential(
                nn.Linear(self.history_periods, d_model),
                nn.ReLU(),
                nn.Linear(d_model, self.future_periods),
            )
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        mean = x_enc.mean(dim=1, keepdim=True).detach()
        centered = x_enc - mean
        batch = centered.shape[0]
        channel_first = centered.transpose(1, 2).reshape(batch * self.enc_in, 1, self.seq_len)
        aggregated = channel_first + self.aggregation(channel_first)
        usable = self.history_periods * self.period
        phases = aggregated[..., -usable:].reshape(batch, self.enc_in, self.history_periods, self.period)
        phases = phases.permute(0, 1, 3, 2)
        future_phases = self.forecaster(phases)
        forecast = future_phases.permute(0, 1, 3, 2).reshape(batch, self.enc_in, -1)
        return forecast[..., : self.pred_len].transpose(1, 2) + mean
