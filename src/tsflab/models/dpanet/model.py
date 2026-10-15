"""Local DPANet: dual temporal/frequency pyramids fused coarse-to-fine by cross-attention.

Independent implementation from the paper (Sections 3.1-3.4, Eqs. 1-6) and the
pinned official code (``hit636/DPANet``). As in the official code, every pyramid
level of every channel is embedded into one ``d_model`` token, so each fusion
block operates on one temporal and one spectral token per channel.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


def log_frequency_bands(length: int, num_scales: int) -> list[tuple[int, int]]:
    """Half-open rfft bin ranges ``[start, end)`` for ``num_scales`` log-spaced bands.

    Band 0 holds the lowest frequencies. Edges are ``floor(exp(u))`` for ``u``
    evenly spaced on ``[0, log F]`` (``F = length // 2 + 1`` bins), with the
    first edge forced to 0 and the last to ``F``; an empty band after the first
    starts where the previous band ended (and then stays empty).
    """
    bins = length // 2 + 1
    if num_scales == 1:
        return [(0, bins)]
    top = math.log(bins) if bins > 1 else 1.0
    edges = np.exp(np.linspace(0.0, top, num_scales + 1)).astype(int)
    edges[0] = 0
    edges[-1] = bins
    bands: list[tuple[int, int]] = []
    for index in range(num_scales):
        start, end = int(edges[index]), int(edges[index + 1])
        if index > 0 and start >= end:
            start = bands[-1][1]
        start = min(start, bins - 1)
        bands.append((start, min(end, bins)))
    return bands


def frequency_pyramid(values: torch.Tensor, bands: list[tuple[int, int]]) -> list[torch.Tensor]:
    """Eq. 2: ``IRFFT(RFFT(x) * M_s)`` per band for ``values`` shaped ``[B, L, C]``."""
    length = values.shape[1]
    spectrum = torch.fft.rfft(values, dim=1)
    levels = []
    for start, end in bands:
        masked = torch.zeros_like(spectrum)
        if start < end:
            masked[:, start:end] = spectrum[:, start:end]
        levels.append(torch.fft.irfft(masked, n=length, dim=1))
    return levels


def temporal_pyramid(values: torch.Tensor, num_scales: int) -> list[torch.Tensor]:
    """Eq. 1: repeated stride-2 average pooling over time of ``[B, L, C]``."""
    levels = [values]
    for _ in range(num_scales - 1):
        pooled = F.avg_pool1d(levels[-1].transpose(1, 2), kernel_size=2, stride=2)
        levels.append(pooled.transpose(1, 2))
    return levels


class CrossPyramidFusionBlock(nn.Module):
    """Eqs. 3-4 plus the fusion FFN on ``[N, tokens, d_model]`` temporal/spectral states.

    The temporal state attends to the spectral state first; the spectral state
    then attends to the updated temporal state (official order). The two are
    concatenated, refined by a residual GELU FFN with post-norm, and split.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.temporal_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.spectral_attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(2 * d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, 2 * d_model),
        )
        self.temporal_norm = nn.LayerNorm(d_model)
        self.spectral_norm = nn.LayerNorm(d_model)
        self.fusion_norm = nn.LayerNorm(2 * d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h_t: torch.Tensor, h_f: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Eq. 3: h_t' = LN(h_t + CrossAttn(h_t, h_f, h_f))
        attended, _ = self.temporal_attention(h_t, h_f, h_f, need_weights=False)
        h_t = self.temporal_norm(h_t + self.dropout(attended))
        # Eq. 4, with the updated temporal state as keys/values (official order).
        attended, _ = self.spectral_attention(h_f, h_t, h_t, need_weights=False)
        h_f = self.spectral_norm(h_f + self.dropout(attended))
        fused = torch.cat([h_t, h_f], dim=-1)
        fused = self.fusion_norm(fused + self.dropout(self.ffn(fused)))
        h_t, h_f = fused.chunk(2, dim=-1)
        return h_t, h_f


class Model(nn.Module):
    """DPANet forecaster for ``[B, seq_len, enc_in]`` histories."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        num_scales: int = 3,
        d_model: int = 128,
        n_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if seq_len // 2 ** (num_scales - 1) < 1:
            raise ValueError("seq_len is too short for num_scales temporal pyramid levels")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_scales = num_scales
        self.d_model = d_model
        self.revin = RevIN(enc_in, eps=1e-5, affine=True)
        self.bands = log_frequency_bands(seq_len, num_scales)
        # Scale-specific embeddings: the whole level of one channel -> one token.
        self.temporal_embedders = nn.ModuleList(nn.Linear(seq_len // 2**scale, d_model) for scale in range(num_scales))
        self.spectral_embedders = nn.ModuleList(nn.Linear(seq_len, d_model) for _ in range(num_scales))
        self.fusion_blocks = nn.ModuleList(
            CrossPyramidFusionBlock(d_model, n_heads, d_ff, dropout) for _ in range(num_scales)
        )
        self.prediction_head = nn.Linear(d_model, pred_len)

    def embed_pyramids(self, values: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Per-level channel tokens ``[B * C, 1, d_model]`` of both pyramids."""
        batch, _, channels = values.shape

        def tokens(levels, embedders):
            out = []
            for level, embed in zip(levels, embedders):
                flat = level.transpose(1, 2).reshape(batch * channels, -1)
                out.append(embed(flat).unsqueeze(1))
            return out

        h_t = tokens(temporal_pyramid(values, self.num_scales), self.temporal_embedders)
        h_f = tokens(frequency_pyramid(values, self.bands), self.spectral_embedders)
        return h_t, h_f

    def fuse(self, h_t: list[torch.Tensor], h_f: list[torch.Tensor]) -> torch.Tensor:
        """Coarse-to-fine fusion (Eqs. 5-6); returns the finest temporal state."""
        top = self.num_scales - 1
        state_t, state_f = self.fusion_blocks[top](h_t[top], h_f[top])
        for scale in range(top - 1, -1, -1):
            # UpSample of a one-token state is a repeat to the (one-token) finer level.
            up_t = state_t.expand(-1, h_t[scale].shape[1], -1)
            up_f = state_f.expand(-1, h_f[scale].shape[1], -1)
            state_t, state_f = self.fusion_blocks[scale](h_t[scale] + up_t, h_f[scale] + up_f)
        return state_t

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        batch = x_enc.shape[0]
        values = self.revin(x_enc, "norm")
        h_t, h_f = self.embed_pyramids(values)
        final = self.fuse(h_t, h_f).squeeze(1)
        forecast = self.prediction_head(final).reshape(batch, self.enc_in, self.pred_len).transpose(1, 2)
        return self.revin(forecast, "denorm")
