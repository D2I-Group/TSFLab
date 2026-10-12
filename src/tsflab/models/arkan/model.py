"""Local AR-KAN: a pre-fitted AR memory module followed by a B-spline KAN.

Paper: AR-KAN: Autoregressive-Weight-Enhanced Kolmogorov-Arnold Network for Time
Series Forecasting (arXiv 2509.02967), Section III-B.

Following the Universal Myopic Mapping Theorem, a bank of fixed linear filters
``h_i(n) = a_i delta(n - i)`` (Eq. 8) built from AR(p) coefficients fitted on the
training series (Eqs. 7, 9-12) feeds a static nonlinear map, a KAN (Eqs. 5-6),
which predicts the next value: ``x_hat(n+1) = KAN(a_0 x(n), ..., a_{p-1} x(n-p+1))``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.bspline_basis import bspline_basis

class BSplineKANLayer(nn.Module):
    """KAN layer ``y_j = sum_i phi_ij(x_i) + b_j`` with ``phi = w_b silu + w_s sum_k c_k B_k``.

    ``B_k`` are B-splines of degree ``order`` on a uniform grid of ``grid``
    intervals over ``[-bound, bound]``, extended by ``order`` knots on each side.
    """

    def __init__(self, in_dim: int, out_dim: int, grid: int = 3, order: int = 3, bound: float = 1.0):
        super().__init__()
        self.in_dim, self.out_dim, self.order = in_dim, out_dim, order
        step = 2.0 * bound / grid
        knots = torch.arange(-order, grid + order + 1, dtype=torch.float32) * step - bound
        self.register_buffer("knots", knots)
        scale = 1.0 / math.sqrt(in_dim)
        self.base_weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-scale, scale))
        self.spline_weight = nn.Parameter(torch.full((out_dim, in_dim), scale))
        self.coefficients = nn.Parameter(torch.randn(out_dim, in_dim, grid + order) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def basis(self, x: torch.Tensor) -> torch.Tensor:
        """Cox-de Boor recursion on the shared knots: ``[N, in]`` -> ``[N, in, grid + order]``."""
        return bspline_basis(x, self.knots, self.order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = F.silu(x) @ self.base_weight.T
        spline = torch.einsum(
            "nik,oik->no", self.basis(x), self.coefficients * self.spline_weight.unsqueeze(-1)
        )
        return base + spline + self.bias


def lagged_moments(segments: torch.Tensor, lag: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Normal-equation sums ``X^T X`` ``[C, p, p]`` and ``X^T y`` ``[C, p]`` of ``[S, T, C]``.

    Rows of ``X`` are length-``lag`` chronological windows and ``y`` the next value.
    """
    windows = segments.unfold(1, lag + 1, 1)  # [S, n, C, lag + 1]
    inputs, target = windows[..., :lag], windows[..., lag]
    return (
        torch.einsum("sncp,sncq->cpq", inputs, inputs),
        torch.einsum("sncp,snc->cp", inputs, target),
    )


def autocorrelation_sums(segments: torch.Tensor, lag: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sums ``sum_n x(n) x(n-i)`` ``[C, lag+1]`` and their counts ``[lag+1]`` (Eq. 12)."""
    length = segments.shape[1]
    sums = torch.stack(
        [(segments[:, i:] * segments[:, : length - i]).sum(dim=(0, 1)) for i in range(lag + 1)],
        dim=-1,
    )
    counts = torch.tensor(
        [segments.shape[0] * (length - i) for i in range(lag + 1)], dtype=segments.dtype
    )
    return sums, counts


def solve_least_squares(xtx: torch.Tensor, xty: torch.Tensor) -> torch.Tensor:
    """Minimum-norm AR(p) least-squares coefficients ``[C, p]`` in chronological order."""
    return (torch.linalg.pinv(xtx) @ xty.unsqueeze(-1)).squeeze(-1)


def solve_yule_walker(sums: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
    """Eqs. (9)-(12): ``a = R^-1 r``, returned in chronological window order ``[C, p]``.

    ``a_i`` multiplies ``x(n - i)``, i.e. window position ``p - 1 - i``.
    """
    r = sums / counts  # [C, p + 1]
    lag = r.shape[-1] - 1
    index = torch.arange(lag)
    toeplitz = r[:, (index.view(-1, 1) - index.view(1, -1)).abs()]  # [C, p, p]
    coefficients = (torch.linalg.pinv(toeplitz) @ r[:, 1:].unsqueeze(-1)).squeeze(-1)
    return coefficients.flip(-1)


class Model(nn.Module):
    """Channel-independent AR-KAN with a recursive one-step-ahead forecast."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        lag: int = 20,
        hidden: int = 50,
        grid: int = 3,
        spline_order: int = 3,
        grid_bound: float = 1.0,
        ar_estimator: str = "least_squares",
    ) -> None:
        super().__init__()
        if lag > seq_len:
            raise ValueError("lag must be between 1 and seq_len")
        self.seq_len, self.pred_len, self.enc_in, self.lag = seq_len, pred_len, enc_in, lag
        self.ar_estimator = ar_estimator
        # AR memory (Eq. 8): one coefficient per lag and channel, fitted before training.
        self.register_buffer("ar_weights", torch.ones(enc_in, lag))
        self.register_buffer("ar_fitted", torch.zeros((), dtype=torch.bool))
        self.kan = nn.Sequential(
            BSplineKANLayer(lag, hidden, grid, spline_order, grid_bound),
            BSplineKANLayer(hidden, 1, grid, spline_order, grid_bound),
        )

    @torch.no_grad()
    def fit_ar(self, series: torch.Tensor, chunk: int = 4096) -> None:
        """Fit the AR(p) memory on a ``[T, C]`` training series (one filter bank per channel)."""
        if series.ndim != 2 or series.shape[1] != self.enc_in or series.shape[0] <= self.lag:
            raise ValueError("series must be [T > lag, enc_in]")
        series = series.double()
        if self.ar_estimator == "yule_walker":
            coefficients = solve_yule_walker(*autocorrelation_sums(series.unsqueeze(0), self.lag))
        else:
            xtx = series.new_zeros(self.enc_in, self.lag, self.lag)
            xty = series.new_zeros(self.enc_in, self.lag)
            for start in range(0, series.shape[0] - self.lag, chunk):
                part = series[start : start + chunk + self.lag].unsqueeze(0)
                moments = lagged_moments(part, self.lag)
                xtx += moments[0]
                xty += moments[1]
            coefficients = solve_least_squares(xtx, xty)
        self.set_ar_weights(coefficients)

    @torch.no_grad()
    def set_ar_weights(self, coefficients: torch.Tensor) -> None:
        self.ar_weights.copy_(coefficients.to(self.ar_weights))
        self.ar_fitted.fill_(True)

    def one_step(self, window: torch.Tensor) -> torch.Tensor:
        """``[B, p, C]`` chronological lags -> ``[B, C]`` next value via ``KAN(a * x)``."""
        batch = window.shape[0]
        weighted = window * self.ar_weights.T.unsqueeze(0)
        tokens = weighted.permute(0, 2, 1).reshape(batch * self.enc_in, self.lag)
        return self.kan(tokens).view(batch, self.enc_in)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("x_enc does not match configured time/channel dimensions")
        window = x_enc[:, -self.lag :, :]
        steps = []
        for _ in range(self.pred_len):
            step = self.one_step(window)
            steps.append(step)
            window = torch.cat([window[:, 1:], step.unsqueeze(1)], dim=1)
        return torch.stack(steps, dim=1)

    def teacher_forced(self, x_enc: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        """One-step predictions of every horizon step from observed lags ``[B, H, C]``."""
        horizon = future.shape[1]
        sequence = torch.cat([x_enc[:, -self.lag :], future[:, :-1]], dim=1)
        windows = sequence.unfold(1, self.lag, 1)  # [B, H, C, lag]
        batch = windows.shape[0]
        flat = windows.permute(0, 1, 3, 2).reshape(batch * horizon, self.lag, self.enc_in)
        return self.one_step(flat).view(batch, horizon, self.enc_in)
