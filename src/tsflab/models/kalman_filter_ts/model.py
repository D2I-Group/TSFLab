"""Independent differentiable fixed-gain alpha-beta forecasting filter."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._inputs import check_window


def _logit(probability: float) -> float:
    return math.log(probability / (1.0 - probability))


class Model(nn.Module):
    """Learn per-channel fixed gains for a constant-velocity state filter."""

    def __init__(
        self, seq_len: int, pred_len: int, enc_in: int, initial_alpha: float = 0.5, initial_beta: float = 0.25
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.alpha_logits = nn.Parameter(torch.full((enc_in,), _logit(initial_alpha)))
        self.beta_logits = nn.Parameter(torch.full((enc_in,), _logit(initial_beta)))
        self.aux_loss: None = None

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        alpha, beta = self.alpha_logits.sigmoid(), self.beta_logits.sigmoid()
        level, velocity = x_enc[:, 0], torch.zeros_like(x_enc[:, 0])
        for index in range(1, self.seq_len):
            predicted_level = level + velocity
            innovation = x_enc[:, index] - predicted_level
            level = predicted_level + alpha * innovation
            velocity = velocity + beta * innovation
        horizon = torch.arange(1, self.pred_len + 1, device=x_enc.device, dtype=x_enc.dtype)
        return level[:, None, :] + horizon[None, :, None] * velocity[:, None, :]
