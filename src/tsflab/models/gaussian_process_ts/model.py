"""Independent sparse RBF-kernel posterior-mean forecasting baseline."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Approximate a zero-mean GP posterior mean with learned inducing pairs."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, num_inducing: int = 16, length_scale: float = 1.0, noise: float = 1e-3) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.inducing_inputs = nn.Parameter(torch.randn(num_inducing, seq_len) * 0.05)
        self.inducing_targets = nn.Parameter(torch.randn(num_inducing, pred_len) * 0.05)
        self.raw_length_scale = nn.Parameter(torch.tensor(math.log(math.expm1(length_scale))))
        self.raw_noise = nn.Parameter(torch.tensor(math.log(math.expm1(noise))))
        self.aux_loss: None = None

    @property
    def length_scale(self) -> torch.Tensor:
        return F.softplus(self.raw_length_scale) + 1e-6

    @property
    def noise(self) -> torch.Tensor:
        return F.softplus(self.raw_noise) + 1e-6

    def _kernel(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * torch.cdist(left, right).square() / self.length_scale.square())

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        queries = x_enc.transpose(1, 2).reshape(-1, self.seq_len)
        k_xz = self._kernel(queries, self.inducing_inputs)
        k_zz = self._kernel(self.inducing_inputs, self.inducing_inputs)
        identity = torch.eye(k_zz.shape[0], device=x_enc.device, dtype=x_enc.dtype)
        coefficients = torch.linalg.solve(k_zz + self.noise * identity, self.inducing_targets)
        forecast = k_xz @ coefficients
        return forecast.reshape(x_enc.shape[0], self.enc_in, self.pred_len).transpose(1, 2)
