"""Independent differentiable conditional ARIMA forecasting baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """Conditional ARIMA(p, 1, q) recurrence shared across channels."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, ar_order: int = 2, ma_order: int = 1) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in) < 1 or ar_order < 0 or ma_order < 0:
            raise ValueError("dimensions must be positive and orders non-negative")
        if ar_order + ma_order < 1:
            raise ValueError("at least one AR or MA coefficient is required")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.ar_order, self.ma_order = ar_order, ma_order
        self.ar_coefficients = nn.Parameter(torch.zeros(ar_order))
        self.ma_coefficients = nn.Parameter(torch.zeros(ma_order))
        self.drift = nn.Parameter(torch.zeros(()))
        self.aux_loss: None = None

    @staticmethod
    def _lag(values: list[torch.Tensor], offset: int, fallback: torch.Tensor) -> torch.Tensor:
        return values[-offset] if len(values) >= offset else fallback

    def _one_step(self, history: list, innovations: list, shift: int, zero: torch.Tensor) -> torch.Tensor:
        """Drift + AR over ``history`` + MA over ``innovations``; MA lags ``<= shift`` (future innovations) drop out."""
        prediction = self.drift + zero
        for lag, coefficient in enumerate(self.ar_coefficients, start=1):
            prediction = prediction + coefficient * self._lag(history, lag, zero)
        for lag, coefficient in enumerate(self.ma_coefficients, start=1):
            if lag > shift:
                prediction = prediction + coefficient * self._lag(innovations, lag - shift, zero)
        return prediction

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}")
        zero = torch.zeros_like(x_enc[:, 0])
        differences = [x_enc[:, index] - x_enc[:, index - 1] for index in range(1, self.seq_len)]
        innovations: list[torch.Tensor] = []
        observed: list[torch.Tensor] = []
        for difference in differences:
            prediction = self._one_step(observed, innovations, 0, zero)
            innovations.append(difference - prediction)
            observed.append(difference)

        future: list[torch.Tensor] = []
        history = observed.copy()
        for _ in range(self.pred_len):
            prediction = self._one_step(history, innovations, len(future), zero)
            future.append(prediction)
            history.append(prediction)
        return x_enc[:, -1:, :] + torch.cumsum(torch.stack(future, dim=1), dim=1)
