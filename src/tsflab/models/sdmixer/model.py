"""Clean-room SDMixer implementation.

Reference: "SDMixer: Sparse Dual-Mixer for Time Series Forecasting"
(arXiv:2602.23581, Section 4 "Method"). Equation numbers in comments below
refer to that section. This is an independent reimplementation from the
paper's equations; no source from the official repository
(https://github.com/SDMixer/SDMixer) was copied. See README.md for the
official-code clarifications that were checked and the points where the
official reference script diverges from (or under-specifies) the paper.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from tsflab.models._components.revin import RevIN


def _topk_mask(scores: torch.Tensor, k: int, dim: int) -> torch.Tensor:
    """Binary mask keeping only the top-``k`` entries of ``scores`` along ``dim``."""
    k = max(1, min(k, scores.shape[dim]))
    _, index = torch.topk(scores, k, dim=dim)
    mask = torch.zeros_like(scores)
    mask.scatter_(dim, index, 1.0)
    return mask


class SpectralDecomposition(nn.Module):
    """Eq. (3)-(5): energy-dominant FFT masking splits ``X`` into season/trend.

    The magnitude spectrum of ``X`` is computed and, independently for every
    (batch, channel) pair, only the ``top_k`` frequency bins with the largest
    amplitude are kept; the corresponding inverse transform is the seasonal
    component, and the trend is the residual ``X - season``.
    """

    def __init__(self, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        length = x.shape[1]
        spectrum = torch.fft.rfft(x, dim=1)
        amplitude = spectrum.abs()
        mask = _topk_mask(amplitude, self.top_k, dim=1)
        season_spectrum = spectrum * mask
        season = torch.fft.irfft(season_spectrum, n=length, dim=1)
        trend = x - season
        return season, trend


class SparseTemporalFlow(nn.Module):
    """Eq. (7)-(10): channel-sparse trend mixer ("Sparse Temporal Flow").

    ``variable_mix`` is ``W_T`` (Eq. 7); ``last_gate`` exposes the magnitude
    gate ``G_T`` (Eq. 8) that keeps only the ``k = ceil(alpha * C)`` channels
    with the largest magnitude at every time step, before the sparsified
    representation is mixed across time by ``temporal_mlp`` (Eq. 10).
    """

    def __init__(
        self,
        channels: int,
        seq_len: int,
        d_ff: int,
        sparse_ratio: float,
        dropout: float,
    ) -> None:
        super().__init__()
        self.variable_mix = nn.Linear(channels, channels)
        self.k = max(1, math.ceil(sparse_ratio * channels))
        self.temporal_mlp = nn.Sequential(
            nn.Linear(seq_len, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, seq_len),
        )
        self.last_gate: torch.Tensor | None = None

    def forward(self, x_trend: torch.Tensor) -> torch.Tensor:
        h_t = self.variable_mix(x_trend)
        gate = _topk_mask(h_t.abs(), self.k, dim=-1)
        self.last_gate = gate
        z_t = h_t * gate
        mixed = self.temporal_mlp(z_t.transpose(1, 2)).transpose(1, 2)
        return mixed


class FrequencyFlow(nn.Module):
    """Eq. (11): amplify the real part of the seasonal spectrum ("Frequency Flow")."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.enhance = nn.Linear(channels, channels)

    def forward(self, x_season: torch.Tensor) -> torch.Tensor:
        length = x_season.shape[1]
        spectrum = torch.fft.rfft(x_season, dim=1)
        enhanced_real = self.enhance(spectrum.real)
        enhanced_spectrum = torch.complex(enhanced_real, spectrum.imag)
        return torch.fft.irfft(enhanced_spectrum, n=length, dim=1)


class SparseCrossMixer(nn.Module):
    """Eq. (12)-(13): trend-conditioned sparse cross-attention ("Sparse Cross-Mixer").

    Query comes from the trend branch, key/value from the frequency branch;
    the softmax attention map is sparsified by keeping only the top
    ``ceil(attn_sparse_ratio * L)`` entries per query row (Eq. 12), and the
    fused output is a sigmoid-gated residual over the trend branch (Eq. 13).
    ``last_attention`` exposes the sparsified map ``alpha`` for structure
    checks.
    """

    def __init__(self, channels: int, sparse_ratio: float) -> None:
        super().__init__()
        self.q_proj = nn.Linear(channels, channels)
        self.k_proj = nn.Linear(channels, channels)
        self.v_proj = nn.Linear(channels, channels)
        self.scale = channels**0.5
        self.sparse_ratio = sparse_ratio
        self.gamma = nn.Parameter(torch.zeros(1))
        self.last_attention: torch.Tensor | None = None

    def forward(self, f_trend: torch.Tensor, f_freq: torch.Tensor) -> torch.Tensor:
        length = f_trend.shape[1]
        query = self.q_proj(f_trend)
        key = self.k_proj(f_freq)
        value = self.v_proj(f_freq)
        logits = torch.matmul(query, key.transpose(-1, -2)) / self.scale
        weights = torch.softmax(logits, dim=-1)
        keep = max(1, math.ceil(self.sparse_ratio * length))
        mask = _topk_mask(weights, keep, dim=-1)
        alpha = weights * mask
        self.last_attention = alpha
        fused = torch.matmul(alpha, value)
        return f_trend + torch.sigmoid(self.gamma) * fused


class Model(nn.Module):
    """SDMixer: dual-stream (time/frequency) sparse Mixer forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        c_out: int,
        spectral_top_k: int = 5,
        channel_sparse_ratio: float = 0.25,
        attn_sparse_ratio: float = 0.5,
        d_ff: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if seq_len < 2:
            raise ValueError("seq_len must be >= 2")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        self.decomposition = SpectralDecomposition(spectral_top_k)
        self.temporal_flow = SparseTemporalFlow(
            enc_in, seq_len, d_ff, channel_sparse_ratio, dropout
        )
        self.frequency_flow = FrequencyFlow(enc_in)
        self.cross_mixer = SparseCrossMixer(enc_in, attn_sparse_ratio)
        self.head = nn.Linear(seq_len, pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.shape[1] != self.seq_len or x_enc.shape[-1] != self.enc_in:
            raise ValueError(
                f"expected input shaped [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )
        x = self.revin(x_enc, "norm")
        season, trend = self.decomposition(x)  # Eq. (3)-(5)
        f_trend = self.temporal_flow(trend)  # Eq. (7)-(10)
        f_freq = self.frequency_flow(season)  # Eq. (11)
        fused = self.cross_mixer(f_trend, f_freq)  # Eq. (12)-(13)
        out = self.head(fused.transpose(1, 2)).transpose(1, 2)  # Eq. (6): project to L'
        out = self.revin(out, "denorm")
        return out
