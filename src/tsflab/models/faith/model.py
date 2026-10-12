"""Paper-driven local implementation of FAITH.

FAITH (Frequency-domain Attention In Two Horizons) splits the RevIN-normalized
window into a trend (learned weighted sum of several edge-padded moving
averages, Eqs. 1-3) and a seasonal residual. The trend is mapped to the
horizon by a linear predictor (Sec. 3.2). The seasonal part is lifted to a
``d_model`` embedding per time step and refined by FCTEM blocks: FCEM applies
self-attention to the rFFT of the channel axis (one sequence per time step)
and FTEM to the rFFT of the time axis (one sequence per channel), both on a
fixed subset of frequency bins, separately on real and imaginary parts, before
the inverse rFFT (Eqs. 7-19). A flatten head forecasts the seasonal part, and
a learned two-weight fusion combines both forecasts (Eqs. 4-5).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.revin import RevIN
from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention
from tsflab.models._components.series_decomposition import EdgePaddedMovingAverage


def select_frequency_bins(pool: int, modes: int, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sorted indices of ``min(modes, pool)`` bins drawn without replacement from ``range(pool)``.

    Eqs. (7)-(8): the selection matrix keeps a fixed number of Fourier components,
    sampled at random rather than truncated to the lowest ones. When ``modes``
    covers the whole pool every bin is kept, in order.
    """
    if pool < 1 or modes < 1:
        raise ValueError("pool and modes must be positive")
    count = min(modes, pool)
    if count == pool:
        return torch.arange(pool)
    return torch.randperm(pool, generator=generator)[:count].sort().values


class MultiScaleTrend(nn.Module):
    """Eqs. (1)-(3): trend = sum_i w_i * MA_{k_i}(x) with learned w (init 1/k); seasonal = x - trend."""

    def __init__(self, kernel_sizes: tuple[int, ...]) -> None:
        super().__init__()
        self.averages = nn.ModuleList(EdgePaddedMovingAverage(k) for k in kernel_sizes)
        self.weight = nn.Parameter(torch.full((len(kernel_sizes),), 1.0 / len(kernel_sizes)))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        stacked = torch.stack([average(x) for average in self.averages])
        trend = torch.einsum("k,kbtn->btn", self.weight, stacked)
        return x - trend, trend


class FrequencyFeatureExtraction(nn.Module):
    """Post-norm self-attention layer with BatchNorm and a GELU feed-forward (PatchTST-style)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = AttentionLayer(
            FullAttention(False, attention_dropout=0.0), d_model, n_heads
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.norm_attention = nn.BatchNorm1d(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model)
        )
        self.ffn_dropout = nn.Dropout(dropout)
        self.norm_ffn = nn.BatchNorm1d(d_model)

    @staticmethod
    def _batch_norm(norm: nn.BatchNorm1d, x: torch.Tensor) -> torch.Tensor:
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x, None)
        x = self._batch_norm(self.norm_attention, x + self.attention_dropout(attended))
        return self._batch_norm(self.norm_ffn, x + self.ffn_dropout(self.feed_forward(x)))


class FrequencyAttention(nn.Module):
    """FEM -> FeatureExtraction -> iFEM along one axis of a ``[S, length, d]`` tensor.

    rFFT over ``length``; the selected bins' real and imaginary parts each pass
    through the same attention layer; unselected bins are zero-padded before
    the inverse rFFT back to ``length`` (Eqs. 10-13).
    """

    def __init__(self, length: int, pool: int, modes: int, d_model: int, n_heads: int,
                 d_ff: int, dropout: float) -> None:
        super().__init__()
        self.length = length
        self.register_buffer("bins", select_frequency_bins(pool, modes))
        self.extraction = FrequencyFeatureExtraction(d_model, n_heads, d_ff, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(x, dim=1)
        selected = spectrum[:, self.bins]
        real = self.extraction(selected.real)
        imag = self.extraction(selected.imag)
        padded = torch.zeros_like(spectrum)
        padded[:, self.bins] = torch.complex(real, imag)
        return torch.fft.irfft(padded, n=self.length, dim=1)


class FCTEMBlock(nn.Module):
    """FCEM over the channel axis per time step, then FTEM over the time axis per channel."""

    def __init__(self, seq_len: int, enc_in: int, modes: int, d_model: int, n_heads: int,
                 d_ff: int, dropout: float) -> None:
        super().__init__()
        # Official pools: (N + 1) // 2 channel bins and L // 2 temporal bins.
        self.fcem = FrequencyAttention(enc_in, (enc_in + 1) // 2, modes, d_model, n_heads, d_ff, dropout)
        self.ftem = FrequencyAttention(seq_len, seq_len // 2, modes, d_model, n_heads, d_ff, dropout)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch, channels, length, width = z.shape  # [B, N, L, d]
        per_step = z.transpose(1, 2).reshape(batch * length, channels, width)
        z = self.fcem(per_step).reshape(batch, length, channels, width).transpose(1, 2)
        per_channel = z.reshape(batch * channels, length, width)
        return self.ftem(per_channel).reshape(batch, channels, length, width)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 32,
        n_heads: int = 4,
        e_layers: int = 1,
        d_ff: int = 256,
        dropout: float = 0.0,
        head_dropout: float = 0.05,
        modes: int = 64,
        kernel_sizes: tuple[int, ...] = (17, 49),
    ) -> None:
        super().__init__()
        if seq_len < 2:
            raise ValueError("seq_len must be at least 2")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = RevIN(enc_in, eps=1e-5, affine=True)
        self.decomposition = MultiScaleTrend(tuple(kernel_sizes))
        # Seasonal embedding: each scalar times one learned d_model token.
        self.token = nn.Parameter(torch.randn(1, d_model))
        self.blocks = nn.ModuleList(
            FCTEMBlock(seq_len, enc_in, modes, d_model, n_heads, d_ff, dropout)
            for _ in range(e_layers)
        )
        self.seasonal_head = FlattenForecastHead(False, enc_in, seq_len * d_model, pred_len, head_dropout)
        self.trend_head = nn.Linear(seq_len, pred_len)
        with torch.no_grad():
            self.trend_head.weight.fill_(1.0 / seq_len)
        self.fusion = nn.Parameter(torch.ones(2))

    def embed(self, seasonal: torch.Tensor) -> torch.Tensor:
        """``[B, L, N] -> [B, N, L, d]``."""
        return seasonal.transpose(1, 2).unsqueeze(-1) * self.token

    def seasonal_forecast(self, seasonal: torch.Tensor) -> torch.Tensor:
        z = self.embed(seasonal)
        for block in self.blocks:
            z = block(z)
        return self.seasonal_head(z).transpose(1, 2)  # [B, H, N]

    def trend_forecast(self, trend: torch.Tensor) -> torch.Tensor:
        return self.trend_head(trend.transpose(1, 2)).transpose(1, 2)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        normalized = self.revin(x_enc, "norm")
        seasonal, trend = self.decomposition(normalized)
        # Eqs. (4)-(5): learned weighted fusion of the seasonal and trend forecasts.
        fused = self.fusion[0] * self.seasonal_forecast(seasonal) + self.fusion[1] * self.trend_forecast(trend)
        return self.revin(fused, "denorm")
