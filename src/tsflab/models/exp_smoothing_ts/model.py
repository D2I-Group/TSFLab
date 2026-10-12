"""Independent differentiable simple-exponential-smoothing baseline."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Learn one smoothing coefficient per channel and repeat the final level."""

    def __init__(
        self, seq_len: int, pred_len: int, enc_in: int, initial_alpha: float = 0.5
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        initial_logit = math.log(initial_alpha / (1.0 - initial_alpha))
        self.alpha_logit = nn.Parameter(torch.full((enc_in,), initial_logit))
        self.aux_loss: torch.Tensor | None = None

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.alpha_logit)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        alpha = self.alpha.view(1, -1)
        level = x_enc[:, 0, :]
        for index in range(1, self.seq_len):
            level = alpha * x_enc[:, index, :] + (1.0 - alpha) * level
        forecast = level.unsqueeze(1).expand(-1, self.pred_len, -1)
        self.aux_loss = forecast.new_zeros(())
        return forecast
