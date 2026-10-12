"""Independent ridge-regularized lag regression baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Channel-independent linear forecast with an L2 training penalty."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, l2_penalty: float = 1e-4) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.l2_penalty = l2_penalty
        self.projection = nn.Linear(seq_len, pred_len)
        self.aux_loss: torch.Tensor | None = None

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        forecast = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        self.aux_loss = self.l2_penalty * self.projection.weight.square().sum()
        return forecast
