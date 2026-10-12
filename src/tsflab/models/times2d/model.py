"""Times2D: multi-period 2D decomposition and derivative heatmaps.

Independent implementation from Section III (Eqs. 1-16, Fig. 1) of Nematirad,
Pahwa and Natarajan, "Times2D: Multi-Period Decomposition and Derivative Mapping
for General Time Series Forecasting" (arXiv 2504.00118, AAAI 2025), after reading
the pinned official code (``Tims2D/Times2D`` at ``ab819a00``, AGPL-3.0 license
file) to resolve omissions; nothing is copied.

Periodic Decomposition Block (PDB): for every period the normalized history is
folded into a ``cycles x period`` grid, cut into full-height column patches by a
2D convolution, encoded by a Transformer over the patch tokens and mapped back to
the window length; a linear head maps the concatenated periods to the horizon.
First and Second Derivative Heatmaps (FSDH): zero-padded first and second
differences are stacked into a ``time x 2`` heatmap, processed by 2D inception
convolutions, summed with learned derivative weights and mapped to the horizon.
The Aggregation Forecasting Block adds both forecasts.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.inception_block import InceptionBlock2d
from tsflab.models._components.positional_encoding import positional_encoding
from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


def derivative_heatmap(x: torch.Tensor) -> torch.Tensor:
    """Eqs. (11)-(14): ``[B, T, N] -> [B, N, T, 2]`` stack of the first difference
    ``D1(t) = x(t) - x(t - 1)`` and second difference ``D2(t) = D1(t) - D1(t - 1)``,
    each left-padded with one zero to keep length ``T``."""
    first = F.pad(x[:, 1:] - x[:, :-1], (0, 0, 1, 0))
    second = F.pad(first[:, 1:] - first[:, :-1], (0, 0, 1, 0))
    return torch.stack((first, second), dim=-1).permute(0, 2, 1, 3)


def period_grid(x: torch.Tensor, period: int, patch_len: int) -> torch.Tensor:
    """Fold ``[B, N, T]`` into ``[B, N, ceil(T / period), cols]``: right zero-pad time
    to a whole number of periods (one row per cycle, one column per phase), then
    right zero-pad the columns to a multiple of ``patch_len``."""
    b, n, t = x.shape
    rows = math.ceil(t / period)
    grid = F.pad(x, (0, rows * period - t)).reshape(b, n, rows, period)
    cols = math.ceil(period / patch_len) * patch_len
    return F.pad(grid, (0, cols - period))


class ResidualAttention(nn.Module):
    """Multi-head self-attention whose pre-softmax scaled scores add the previous
    layer's scores (residual attention) and are passed on."""

    def __init__(self, d_model: int, n_heads: int, attn_dropout: float, proj_dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

    def forward(
        self, x: torch.Tensor, prev: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, length, _ = x.shape
        q = self.query(x).view(b, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(b, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(b, length, self.n_heads, self.head_dim).transpose(1, 2)
        scores = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if prev is not None:
            scores = scores + prev
        attn = self.attn_dropout(torch.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, length, -1)
        return self.proj_dropout(self.out(out)), scores


class BatchNormTokens(nn.Module):
    """BatchNorm1d over the feature axis of ``[B, L, D]`` tokens."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class TokenEncoderLayer(nn.Module):
    """Eqs. (6)-(9): post-norm layer, residual attention + BatchNorm, GELU
    feed-forward + BatchNorm, dropout on both residual branches."""

    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, attn_dropout: float
    ) -> None:
        super().__init__()
        self.attention = ResidualAttention(d_model, n_heads, attn_dropout, dropout)
        self.norm_attn = BatchNormTokens(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.norm_ffn = BatchNormTokens(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, prev: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attended, scores = self.attention(x, prev)
        x = self.norm_attn(x + self.dropout(attended))
        x = self.norm_ffn(x + self.dropout(self.ffn(x)))
        return x, scores


class PeriodBranch(nn.Module):
    """PDB for one period (Eqs. 4-10): grid, full-height ``rows x patch_len`` patch
    convolution, linear token embedding plus a learnable position table, token
    encoder, and a flatten-linear map back to ``seq_len``."""

    def __init__(
        self,
        seq_len: int,
        period: int,
        patch_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        e_layers: int,
        dropout: float,
        attn_dropout: float,
    ) -> None:
        super().__init__()
        self.period = period
        self.patch_len = patch_len
        self.rows = math.ceil(seq_len / period)
        self.tokens = math.ceil(period / patch_len)
        patch_dim = self.rows * patch_len
        self.patch_conv = nn.Conv2d(1, patch_dim, kernel_size=(self.rows, patch_len), stride=(self.rows, patch_len))
        self.embed = nn.Linear(patch_dim, d_model)
        self.position = positional_encoding("zeros", True, self.tokens, d_model)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            TokenEncoderLayer(d_model, n_heads, d_ff, dropout, attn_dropout) for _ in range(e_layers)
        )
        width = self.tokens * d_model
        self.to_window = nn.Linear(width, seq_len) if width != seq_len else nn.Identity()

    def patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, N, T] -> [B * N, tokens, rows * patch_len]`` conv patch features."""
        b, n, _ = x.shape
        grid = period_grid(x, self.period, self.patch_len).reshape(b * n, 1, self.rows, -1)
        return self.patch_conv(grid).flatten(2).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        h = self.dropout(self.embed(self.patch_tokens(x)) + self.position)
        scores = None
        for layer in self.layers:
            h, scores = layer(h, scores)
        # Flatten feature-major ([d_model, tokens]) as in the official head.
        h = h.transpose(1, 2).reshape(b, n, -1)
        return self.to_window(h)


class Model(nn.Module):
    """Times2D forecaster ``[B, seq_len, enc_in] -> [B, pred_len, enc_in]``; marks
    and decoder inputs are ignored."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        periods: tuple[int, ...] | list[int] = (720, 360, 110, 96, 48),
        patch_lens: tuple[int, ...] | list[int] = (48, 32, 16, 6, 3),
        d_model: int = 64,
        n_heads: int = 16,
        e_layers: int = 3,
        d_ff: int = 64,
        num_kernels: int = 6,
        dropout: float = 0.5,
        attn_dropout: float = 0.05,
        head_dropout: float = 0.0,
        affine: bool = False,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        periods, patch_lens = list(periods), list(patch_lens)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = RevIN(enc_in, affine=affine, subtract_last=subtract_last)
        # FSDH (Eqs. 11-16)
        self.heatmap_conv = nn.Sequential(
            InceptionBlock2d(enc_in, d_ff, num_kernels),
            nn.GELU(),
            InceptionBlock2d(d_ff, enc_in, num_kernels),
        )
        self.derivative_weights = nn.Parameter(torch.randn(2))
        self.heatmap_to_horizon = nn.Linear(seq_len, pred_len)
        # PDB (Eqs. 1-10)
        self.branches = nn.ModuleList(
            PeriodBranch(seq_len, p, k, d_model, n_heads, d_ff, e_layers, dropout, attn_dropout)
            for p, k in zip(periods, patch_lens)
        )
        self.head = nn.Linear(seq_len * len(periods), pred_len)
        self.head_dropout = nn.Dropout(head_dropout)

    def fsdh(self, x: torch.Tensor) -> torch.Tensor:
        """Derivative-heatmap forecast ``[B, N, pred_len]`` from normalized ``[B, T, N]``."""
        features = self.heatmap_conv(derivative_heatmap(x))  # [B, N, T, 2]
        summed = (features * self.derivative_weights).sum(-1)  # [B, N, T]
        return self.heatmap_to_horizon(summed)

    def pdb(self, x: torch.Tensor) -> torch.Tensor:
        """Multi-period forecast ``[B, N, pred_len]`` from normalized ``[B, T, N]``."""
        series = x.transpose(1, 2)
        windows = torch.cat([branch(series) for branch in self.branches], dim=-1)
        return self.head_dropout(self.head(windows))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.enc_in)
        x = self.revin(x_enc, "norm")
        combined = self.pdb(x) + self.fsdh(x)  # Eq. (13)
        return self.revin(combined.transpose(1, 2), "denorm")
