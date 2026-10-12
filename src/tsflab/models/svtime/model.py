"""Independent SVTime implementation from the paper's defining equations.

This package implements the paper's ``SVTime`` (not ``SVTime-t``) route:
within-period patches have separate learned inter-period maps (IB1/IB2), and a
backcast residual is projected as a trend and combined through a scalar gate.
The distance-attenuating annealing constraint is specific to SVTime-t and is
therefore deliberately not attributed to this named model.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class PatchWisePeriodMap(nn.Module):
    """Apply an independent period-to-period linear map within each patch."""

    def __init__(
        self,
        input_periods: int,
        output_periods: int,
        period_len: int,
        patch_size: int,
    ) -> None:
        super().__init__()
        self.input_periods = input_periods
        self.output_periods = output_periods
        self.period_len = period_len
        self.patch_size = patch_size
        self.patch_count = math.ceil(period_len / patch_size)
        self.weight = nn.Parameter(
            torch.empty(self.patch_count, input_periods, output_periods)
        )
        nn.init.xavier_uniform_(self.weight)

    def forward(self, periods: torch.Tensor) -> torch.Tensor:
        if periods.ndim != 3 or periods.shape[1:] != (
            self.input_periods,
            self.period_len,
        ):
            raise ValueError(
                "PatchWisePeriodMap expects "
                f"(batch, {self.input_periods}, {self.period_len})"
            )
        output = periods.new_empty(
            periods.shape[0], self.output_periods, self.period_len
        )
        for patch in range(self.patch_count):
            start = patch * self.patch_size
            stop = min(start + self.patch_size, self.period_len)
            values = periods[:, :, start:stop]
            mapped = torch.einsum("bnp,no->bop", values, self.weight[patch])
            output[:, :, start:stop] = mapped
        return output


class SVTimeModel(nn.Module):
    """SVTime with IB1/IB2 and the paper's backcast-residual Eq. (3)."""

    def __init__(
        self,
        c_in: int,
        period: int,
        seq_len: int,
        pred_len: int,
        patch_size: int,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        if period > seq_len:
            raise ValueError("period must be in [1, seq_len]")
        self.c_in = c_in
        self.period = period
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.revin = revin
        self.revin_layer = (
            RevIN(c_in, affine=affine, subtract_last=subtract_last)
            if revin
            else None
        )

        self.history_periods = seq_len // period
        self.used_history = self.history_periods * period
        self.future_periods = math.ceil(pred_len / period)
        self.period_map = PatchWisePeriodMap(
            self.history_periods,
            self.history_periods + self.future_periods,
            period,
            patch_size,
        )
        self.trend_projection = nn.Linear(
            self.used_history, self.future_periods * period
        )
        self.trend_gate_logit = nn.Parameter(torch.zeros(()))

    def period_backcast_forecast(
        self, history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return seasonal backcast/forecast before the trend correction."""
        periods = history.reshape(-1, self.history_periods, self.period)
        mapped = self.period_map(periods)
        return (
            mapped[:, : self.history_periods],
            mapped[:, self.history_periods :],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        check_window(x, self.seq_len, self.c_in)
        if self.revin_layer is not None:
            x = self.revin_layer(x, "norm")

        batch = x.shape[0]
        history = x[:, -self.used_history :, :].permute(0, 2, 1)
        history = history.reshape(batch * self.c_in, self.used_history)
        backcast_periods, forecast_periods = self.period_backcast_forecast(history)
        backcast = backcast_periods.reshape(batch * self.c_in, self.used_history)
        seasonal = forecast_periods.reshape(
            batch * self.c_in, self.future_periods * self.period
        )

        residual = history - backcast
        trend = self.trend_projection(residual)
        gate = torch.sigmoid(self.trend_gate_logit)
        forecast = gate * trend + (1.0 - gate) * seasonal
        forecast = forecast[:, : self.pred_len]
        forecast = forecast.reshape(batch, self.c_in, self.pred_len).permute(0, 2, 1)

        if self.revin_layer is not None:
            forecast = self.revin_layer(forecast, "denorm")
        return forecast


class Model(nn.Module):
    def __init__(
        self,
        c_in: int,
        period: int,
        seq_len: int,
        pred_len: int,
        patch_size: int,
        revin: bool,
        affine: bool,
        subtract_last: bool,
    ) -> None:
        super().__init__()
        self.model = SVTimeModel(
            c_in=c_in,
            period=period,
            seq_len=seq_len,
            pred_len=pred_len,
            patch_size=patch_size,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
        )

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        return self.model(x_enc)
