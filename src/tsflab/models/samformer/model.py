"""SAMformer: one shallow channel-wise attention block plus a linear forecaster.

Per series ``x`` (``[batch, seq_len, channels]``, transposed so each channel is
a token whose features are its ``seq_len`` time steps):

    x_n  = RevIN(x)                                  # per-instance mean/std
    A    = softmax(Q K^T / sqrt(hid_dim)), Q = W_q x_n, K = W_k x_n
    z    = x_n + A (W_v x_n)                         # residual attention
    y    = RevIN^-1(W_f z)                           # linear forecaster, denorm

``Q``/``K`` project time to ``hid_dim``; ``V`` keeps width ``seq_len`` so the
residual add is shape-preserving. Sharpness-aware minimization is a training
procedure (see ``spec.training_objective`` and the ``sharpness_aware``
component); ``forward`` is the plain network.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._components.channel_wise_linear import ChannelWiseLinear
from tsflab.models._components.revin import RevIN
from tsflab.models._components.sharpness_aware import sharpness_aware_loss
from tsflab.models._inputs import check_window


class ChannelWiseAttention(nn.Module):
    """Single-head attention across channels with temporal-axis features."""

    def __init__(self, seq_len: int, hid_dim: int) -> None:
        super().__init__()
        self.scale = 1.0 / math.sqrt(hid_dim)
        self.query = nn.Linear(seq_len, hid_dim)
        self.key = nn.Linear(seq_len, hid_dim)
        self.value = nn.Linear(seq_len, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[batch, channels, seq_len]``; returns the same shape."""
        weights = torch.softmax(
            self.query(x) @ self.key(x).transpose(-2, -1) * self.scale, dim=-1
        )
        return weights @ self.value(x)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hid_dim: int = 16,
        rho: float = 0.5,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.rho = float(rho)  # SAM radius; used only by the training objective
        self.revin = RevIN(enc_in, affine=True, enabled=use_revin)
        self.attention = ChannelWiseAttention(seq_len, hid_dim)
        self.forecaster = ChannelWiseLinear(seq_len, pred_len, enc_in, individual=False)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        x = self.revin(x_enc, "norm").transpose(1, 2)  # [B, C, L]
        x = x + self.attention(x)
        y = self.forecaster(x).transpose(1, 2)  # [B, H, C]
        return self.revin(y, "denorm")

    def sharpness_aware_objective(self, batch, criterion):
        """SAM loss ``criterion`` at ``w + rho g/||g||``; returns ``(forecast, loss)``."""

        def loss_fn(run):
            outputs = run(batch.x, batch.x_mark, batch.dec_inp, batch.y_mark)
            return outputs, criterion(batch.align(outputs), batch.target)

        return sharpness_aware_loss(self, loss_fn, self.rho)
