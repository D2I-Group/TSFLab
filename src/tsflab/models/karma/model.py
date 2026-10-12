"""KARMA: adaptive seasonal-trend decomposition and hybrid frequency-temporal
Mamba blocks over inverted variate tokens.

Independent implementation from Section 3 (Eqs. 2-12, 15, Fig. 1) of Ye et al.,
"KARMA: A Multilevel Decomposition Hybrid Mamba Framework for Multivariate
Long-Term Time Series Forecasting" (arXiv 2506.08939, WASA 2025), after reading
the pinned official code (``yedadasd/KARMA`` at ``456d28f5``, no license file) to
resolve omissions; nothing is copied. The selective SSM is the cataloged
pure-PyTorch ``mamba`` block instead of the ``mamba_ssm`` kernels, and the db4
wavelet pair is the cataloged ``orthogonal_dwt`` (``symmetric`` mode) instead of
``pytorch_wavelets``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.embed import DataEmbedding_inverted
from tsflab.models._components.mamba import MambaBlock, RMSNorm
from tsflab.models._components.marks import encoder_timef_marks, tslib_time_feature_dimension
from tsflab.models._components.orthogonal_dwt import OrthogonalDWT
from tsflab.models._components.revin import RevIN
from tsflab.models._components.series_decomposition import SeriesDecomposition
from tsflab.models._inputs import check_window

#: ATCD constants resolved from the official code: four attention heads, dropout 0.1.
ATCD_HEADS = 4
ATCD_DROPOUT = 0.1


def _scan(d_model: int, d_state: int, d_conv: int, expand: int) -> MambaBlock:
    return MambaBlock(
        d_model=d_model,
        d_inner=expand * d_model,
        dt_rank=math.ceil(d_model / 16),
        d_conv=d_conv,
        d_state=d_state,
        reference_dt_init=True,
    )


class AdaptiveTimeChannelDecomposition(nn.Module):
    """ATCD (Section 3.3): project channels (Eq. 3), extract the trend with
    multi-head self-attention over time and SiLU, take the seasonal part as the
    remainder (Eqs. 4-5), and project both back to the channels (Eq. 6)."""

    def __init__(self, enc_in: int, embed_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(enc_in, embed_dim)
        self.dropout = nn.Dropout(ATCD_DROPOUT)
        self.attention = nn.MultiheadAttention(embed_dim, ATCD_HEADS, batch_first=True)
        self.seasonal_projection = nn.Linear(embed_dim, enc_in)
        self.trend_projection = nn.Linear(embed_dim, enc_in)

    def raw_components(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(X~_s, X~_t)`` in the projected space, ``[B, L, embed_dim]`` each."""
        projected = self.dropout(self.input_projection(x))
        attended, _ = self.attention(projected, projected, projected, need_weights=False)
        trend = F.silu(attended)
        return projected - trend, trend

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seasonal, trend = self.raw_components(x)
        return self.seasonal_projection(seasonal), self.trend_projection(trend)


class KarmaBlock(nn.Module):
    """One KarmaBlock (Eqs. 11-12): low- and high-frequency Mambas on the wavelet
    coefficients, and a bidirectional temporal Mamba with RMSNorm and residual
    on the token sequence."""

    def __init__(
        self, d_model: int, freq_width: int, d_state: int, d_conv: int, expand: int, share_temporal: bool
    ) -> None:
        super().__init__()
        self.low_mamba = _scan(freq_width, d_state, d_conv, expand)
        self.high_mamba = _scan(freq_width, d_state, d_conv, expand)
        self.temporal_mamba = _scan(d_model, d_state, d_conv, expand)
        self.reverse_mamba = self.temporal_mamba if share_temporal else _scan(d_model, d_state, d_conv, expand)
        self.forward_norm = RMSNorm(d_model)
        self.reverse_norm = RMSNorm(d_model)

    def forward(
        self, tokens: torch.Tensor, low: torch.Tensor, high: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        low, high = self.low_mamba(low), self.high_mamba(high)
        reverse = self.reverse_mamba(self.reverse_norm(tokens).flip(1)).flip(1)
        tokens = self.temporal_mamba(self.forward_norm(tokens)) + reverse + tokens
        return tokens, low, high


class HybridFrequencyTemporalEncoder(nn.Module):
    """HFTD with stacked KarmaBlocks (Section 3.4): db4 DWT of every token along
    its embedding axis (Eqs. 7-9), KarmaBlocks on the frequency and temporal
    streams, inverse DWT plus the temporal stream (Eq. 15), then LayerNorm."""

    def __init__(
        self, d_model: int, e_layers: int, d_state: int, d_conv: int, expand: int, share_temporal: bool
    ) -> None:
        super().__init__()
        # Single-level db4 with half-sample symmetric extension (the ``pywt`` /
        # ``pytorch_wavelets`` ``"symmetric"`` mode); filters are not checkpointed.
        self.wavelet = OrthogonalDWT("db4", levels=1, mode="symmetric", persistent=False)
        freq_width = self.wavelet.coefficient_length(d_model)
        self.blocks = nn.ModuleList(
            KarmaBlock(d_model, freq_width, d_state, d_conv, expand, share_temporal) for _ in range(e_layers)
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        low, high = self.wavelet.analyze(tokens)
        for block in self.blocks:
            tokens, low, high = block(tokens, low, high)
        return self.norm(self.wavelet.synthesize(low, high) + tokens)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 512,
        e_layers: int = 2,
        d_state: int = 32,
        d_conv: int = 4,
        expand: int = 2,
        embed_dim: int = 128,
        decomposition: str = "atcd",
        moving_avg: int = 25,
        use_decomp: bool = True,
        dropout: float = 0.1,
        use_norm: bool = True,
        use_marks: bool = True,
        freq: str = "h",
        time_loss_weight: float = 0.2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.use_decomp = use_decomp
        self.use_marks = use_marks
        self.freq = freq
        self.time_loss_weight = time_loss_weight
        if use_marks and not use_decomp:
            tslib_time_feature_dimension(freq)  # validates the calendar layout early
        self.normalizer = RevIN(enc_in, eps=1e-5, affine=True, enabled=use_norm)
        self.seasonal_encoder = HybridFrequencyTemporalEncoder(
            d_model, e_layers, d_state, d_conv, expand, share_temporal=use_decomp
        )
        if use_decomp:
            self.decomposer: nn.Module = (
                AdaptiveTimeChannelDecomposition(enc_in, embed_dim)
                if decomposition == "atcd"
                else SeriesDecomposition(moving_avg)
            )
            self.seasonal_embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
            self.trend_embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
            self.global_mamba = _scan(d_model, d_state, d_conv, expand)
        else:
            self.embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)
        self.projection = nn.Linear(d_model, pred_len)

    def calendar_tokens(self, x_mark_enc: torch.Tensor | None) -> torch.Tensor | None:
        return encoder_timef_marks(x_mark_enc, seq_len=self.seq_len, freq=self.freq, enabled=self.use_marks)

    def encode(self, x: torch.Tensor, x_mark_enc: torch.Tensor | None = None) -> torch.Tensor:
        """Normalized window to fused token representations ``[B, N(+marks), d_model]``."""
        if not self.use_decomp:
            return self.seasonal_encoder(self.embedding(x, self.calendar_tokens(x_mark_enc)))
        seasonal, trend = self.decomposer(x)
        seasonal_tokens = self.seasonal_encoder(self.seasonal_embedding(seasonal, None))
        trend_tokens = self.global_mamba(self.trend_embedding(trend, None))
        return seasonal_tokens + trend_tokens

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        x = self.normalizer(x_enc, "norm")
        forecast = self.projection(self.encode(x, x_mark_enc)).transpose(1, 2)[:, :, : self.enc_in]
        return self.normalizer(forecast, "denorm")

    def hybrid_loss(self, prediction: torch.Tensor, target: torch.Tensor, time_loss: torch.Tensor) -> torch.Tensor:
        """Eq. (2): ``alpha * time-domain loss + (1 - alpha) * mean |rfft(y) - rfft(y_hat)|``
        over the horizon axis (the official ``htfLoss`` with complex rfft residuals)."""
        spectral = (torch.fft.rfft(prediction, dim=1) - torch.fft.rfft(target, dim=1)).abs().mean()
        return self.time_loss_weight * time_loss + (1.0 - self.time_loss_weight) * spectral
