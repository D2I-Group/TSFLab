"""TFKAN: dual-branch time-frequency Kolmogorov-Arnold network forecaster.

Independent implementation from Section III (Eqs. 1-13) of Kui, Liu, Li, Hu, Shi,
Si and Zou, "TFKAN: Time-Frequency KAN for Long-Term Time Series Forecasting"
(arXiv 2506.12696), after reading the pinned official code (``LcWave/TFKAN`` at
``d787829c``, no license file) to resolve omissions; nothing is copied.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.bspline_basis import bspline_basis
from tsflab.models._inputs import check_window


class BSplineKANLinear(nn.Module):
    """One KAN layer ``z = W_base SiLU(x) + sum_i c_i B_i^k(x)`` (Eqs. 1-4).

    Every input feature has its own uniform knot vector extended by ``k`` knots on
    each side of ``grid_range`` (``s + 2k + 1`` knots, ``s + k`` bases). The spline
    weight is ``spline_weight * spline_scaler`` (per-edge coefficients times a
    per-edge scale); there is no output bias. Inputs outside the extended grid
    only pass through the base branch.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 2,
        spline_order: int = 1,
        scale_noise: float = 0.1,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.grid_size, self.spline_order = grid_size, spline_order
        step = (grid_range[1] - grid_range[0]) / grid_size
        knots = torch.arange(-spline_order, grid_size + spline_order + 1, dtype=torch.float32)
        knots = knots * step + grid_range[0]
        self.register_buffer("grid", knots.expand(in_features, -1).contiguous())
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, grid_size + spline_order))
        self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))
        self.reset_parameters(scale_noise)

    @torch.no_grad()
    def reset_parameters(self, scale_noise: float) -> None:
        """Official initialization: Kaiming-uniform base weight and spline scale; spline
        coefficients fitted by least squares to small uniform noise at the interior knots."""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5))
        interior = self.grid[:, self.spline_order : self.grid.shape[1] - self.spline_order]  # [in, s+1]
        noise = torch.rand(self.in_features, interior.shape[1], self.out_features) - 0.5
        noise = noise * scale_noise / self.grid_size
        basis = self.b_splines(interior.T)  # [s+1, in, s+k]
        coeff = torch.linalg.lstsq(basis.transpose(0, 1), noise).solution  # [in, s+k, out]
        self.spline_weight.copy_(coeff.permute(2, 0, 1))

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Cox-de Boor recursion (Eq. 3) on per-input knots: ``[n, in] -> [n, in, s + k]``."""
        return bspline_basis(x, self.grid, self.spline_order)

    def scaled_spline_weight(self) -> torch.Tensor:
        return self.spline_weight * self.spline_scaler.unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, self.in_features)
        base = F.linear(F.silu(flat), self.base_weight)  # Eq. (1)
        spline = F.linear(  # Eq. (2)
            self.b_splines(flat).flatten(1), self.scaled_spline_weight().flatten(1)
        )
        return (base + spline).reshape(*shape[:-1], self.out_features)  # Eq. (4)


class KAN(nn.Sequential):
    """Stack of :class:`BSplineKANLinear` layers with widths ``[in, hidden..., out]``."""

    def __init__(self, widths: list[int], grid_size: int, spline_order: int, scale_noise: float) -> None:
        super().__init__(
            *(
                BSplineKANLinear(a, b, grid_size=grid_size, spline_order=spline_order, scale_noise=scale_noise)
                for a, b in zip(widths[:-1], widths[1:])
            )
        )


class Model(nn.Module):
    """TFKAN forecaster over ``[batch, seq_len, enc_in]``; channels are processed independently."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        embed_size: int = 128,
        hidden_size: int = 256,
        grid_size: int = 2,
        spline_order: int = 1,
        scale_noise: float = 0.1,
        sparsity_threshold: float = 0.001,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.sparsity_threshold = sparsity_threshold
        self.use_bias = use_bias
        self.embedding = nn.Parameter(torch.randn(1, embed_size))  # W in Eq. (5)
        kan = dict(grid_size=grid_size, spline_order=spline_order, scale_noise=scale_noise)
        self.freq_kan = KAN([embed_size, hidden_size, embed_size], **kan)  # FreqKAN, Eq. (8)
        self.time_kan = KAN([seq_len, hidden_size, seq_len], **kan)  # TimeKAN, Eq. (10)
        self.predictor = KAN([seq_len * embed_size, hidden_size, pred_len], **kan)  # Eq. (12)

    def adjust_dimension(self, series: torch.Tensor) -> torch.Tensor:
        """Eq. (5): ``F_t = X_t x W`` lifts ``[B, N, L]`` to ``[B, N, L, d]``; the time branch keeps ``X_t``."""
        return series.unsqueeze(-1) * self.embedding

    def frequency_branch(self, lifted: torch.Tensor) -> torch.Tensor:
        """Eqs. (6)-(9): rFFT over L, one shared KAN on real and imaginary parts, soft-shrink, inverse rFFT."""
        spectrum = torch.fft.rfft(lifted, dim=2, norm="ortho")
        parts = torch.stack((self.freq_kan(spectrum.real), self.freq_kan(spectrum.imag)), dim=-1)
        parts = F.softshrink(parts, lambd=self.sparsity_threshold)
        return torch.fft.irfft(torch.view_as_complex(parts), n=self.seq_len, dim=2, norm="ortho")

    def time_branch(self, series: torch.Tensor) -> torch.Tensor:
        """Eq. (10): TimeKAN maps each channel's length-L history to length L."""
        return self.time_kan(series)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        batch = x_enc.shape[0]
        series = x_enc.transpose(1, 2)  # T_t: [B, N, L]
        lifted = self.adjust_dimension(series)  # F_t: [B, N, L, d]
        hidden = self.frequency_branch(lifted) + self.time_branch(series).unsqueeze(-1)
        if self.use_bias:  # Eq. (11): BIAS = F_t + T_t (broadcast over d)
            hidden = hidden + lifted + series.unsqueeze(-1)
        out = self.predictor(hidden.reshape(batch, self.enc_in, -1))  # [B, N, tau]
        return out.transpose(1, 2)
