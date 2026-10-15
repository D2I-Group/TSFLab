"""HADL: Haar DWT, DCT and a low-rank layer for noise-resilient long-term forecasting.

Independent implementation of arXiv 2502.10569 (Sec. 3, Eqs. 1-5, Algorithm 1),
with omissions resolved from the official repository at revision
``aecbda0dafd7c1ab6bcdb2d3b0e1f0f053f8e30b`` (``models/HADL.py``,
``exp/exp_main.py``). Nothing is copied from it.

Pipeline per channel (paper notation, input ``X_T`` in ``R^{C x L}``):

    X_T  <- X_T - mean_t(X_T)                (official code, restored at the end)
    A_T, _ = DWT_haar(X_T)                   (Eq. 1a / Eq. 2, detail discarded)
    A_F  = (2 / L) * DCT-II(A_T)             (Eq. 1b / Eq. 4, orthonormal DCT)
    Y_T  = A_F P Q + B                        (Eq. 1c / Eq. 5, low-rank layer)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.haar_dwt1d import HaarDWT1D
from tsflab.models._inputs import check_window


def dct_ii_matrix(length: int) -> torch.Tensor:
    """Orthonormal DCT-II matrix ``D`` with ``X = D x`` along the last axis.

    ``D[k, n] = s_k cos(pi k (2n + 1) / (2N))`` with ``s_0 = sqrt(1/N)`` and
    ``s_k = sqrt(2/N)`` otherwise, so ``D`` is orthogonal (Parseval, Eq. 7).
    """
    if length < 1:
        raise ValueError("DCT length must be positive")
    n = torch.arange(length, dtype=torch.float64)
    k = n.unsqueeze(1)
    basis = torch.cos(math.pi * k * (2.0 * n + 1.0) / (2.0 * length))
    scale = torch.full((length, 1), math.sqrt(2.0 / length), dtype=torch.float64)
    scale[0, 0] = math.sqrt(1.0 / length)
    return (scale * basis).to(torch.float32)


class LowRankLinear(nn.Module):
    """``y = x P Q + B`` with ``P: [in, rank]``, ``Q: [rank, out]`` (Eq. 5)."""

    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features, self.out_features, self.rank = in_features, out_features, rank
        # Official initialization: Kaiming-uniform on both factors as stored
        # ([in, rank] and [rank, out]) and a U(0, 1) bias.
        self.P = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(in_features, rank)))
        self.Q = nn.Parameter(nn.init.kaiming_uniform_(torch.empty(rank, out_features)))
        self.B = nn.Parameter(nn.init.uniform_(torch.empty(out_features))) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = (x @ self.P) @ self.Q
        return out if self.B is None else out + self.B


def _check_config(seq_len: int, enable_haar: bool) -> None:
    if seq_len < 2 and enable_haar:
        raise ValueError("the Haar step needs seq_len >= 2")


class Model(nn.Module):
    """HADL forecaster for ``[batch, seq_len, enc_in]`` inputs."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        rank: int = 50,
        bias: bool = True,
        individual: bool = False,
        enable_haar: bool = True,
        enable_dct: bool = True,
        enable_lowrank: bool = True,
        regularization_rate: float = 0.1,
    ) -> None:
        super().__init__()
        _check_config(seq_len, enable_haar)
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.individual = bool(individual)
        self.enable_haar, self.enable_dct = bool(enable_haar), bool(enable_dct)
        self.enable_lowrank = bool(enable_lowrank)
        self.regularization_rate = float(regularization_rate)
        self.in_len = (seq_len + 1) // 2 if self.enable_haar else seq_len
        self.haar = HaarDWT1D()
        if self.enable_dct:
            self.register_buffer("dct_matrix", dct_ii_matrix(self.in_len), persistent=False)

        def make_layer() -> nn.Module:
            if self.enable_lowrank:
                return LowRankLinear(self.in_len, pred_len, rank, bias=bias)
            return nn.Linear(self.in_len, pred_len, bias=bias)

        self.pred_layer = nn.ModuleList(make_layer() for _ in range(enc_in)) if self.individual else make_layer()
        self.aux_loss: torch.Tensor | None = None

    def haar_approximation(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. 2: keep the Haar approximation ``(x_2i + x_2i+1) / sqrt(2)``.

        Odd lengths are zero-padded at the end (zero is the channel mean after
        centering), as in the official code; the shared component would
        otherwise replicate the last sample.
        """
        if x.shape[-1] % 2:
            x = F.pad(x, (0, 1))
        approx, _ = self.haar(x)
        return approx

    def frequency_features(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. 4: ``A_F = (2 / L) DCT(A_T)``, i.e. orthonormal DCT-II over its own length."""
        return (x @ self.dct_matrix.T) / x.shape[-1]

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        if self.individual:
            return torch.stack([layer(x[:, i]) for i, layer in enumerate(self.pred_layer)], dim=1)
        return self.pred_layer(x)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        x = x_enc.transpose(1, 2)  # [B, C, L]
        mean = x.mean(dim=-1, keepdim=True)
        x = x - mean
        if self.enable_haar:
            x = self.haar_approximation(x)
        if self.enable_dct:
            x = self.frequency_features(x)
        out = self._predict(x)
        out = out + mean
        forecast = out.transpose(1, 2)
        # Official L1 regularizer: rate * mean |forecast| added to the training loss.
        if self.training and self.regularization_rate > 0:
            self.aux_loss = self.regularization_rate * forecast.abs().mean()
        else:
            self.aux_loss = None
        return forecast
