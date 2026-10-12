"""Independent differentiable Bayesian-ridge MAP forecasting baseline."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Shared channel-wise lag regression with a learned Gaussian prior precision.

    This is a MAP adaptation of Bayesian ridge regression. It deliberately does
    not claim evidence maximisation or posterior-predictive uncertainty.
    """

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, initial_weight_precision: float = 1e-3) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.projection = nn.Linear(seq_len, pred_len)
        self.log_weight_precision = nn.Parameter(
            torch.tensor(math.log(math.expm1(initial_weight_precision)))
        )
        self.aux_loss: torch.Tensor | None = None

    @property
    def weight_precision(self) -> torch.Tensor:
        return F.softplus(self.log_weight_precision) + 1e-8

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        forecast = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        precision, weights = self.weight_precision, self.projection.weight
        self.aux_loss = 0.5 * precision * weights.square().sum()
        self.aux_loss = self.aux_loss - 0.5 * weights.numel() * precision.log()
        return forecast
