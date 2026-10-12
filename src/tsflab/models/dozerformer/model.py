"""Dozerformer: patch Transformer with Dozer (Local, Stride, Vary) sparse attention.

Independent implementation of Zhang et al., "Sparse Transformer with Local and
Seasonal Adaptation for Multivariate Time Series Forecasting" (arXiv 2312.06874,
Scientific Reports 2024), checked against GRYGY1215/Dozerformer@7598cf97.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._components.series_decomposition import EdgePaddedMovingAverage

ScoreMode = Literal["zero", "masked"]


def dozer_self_mask(num_patches: int, local_window: int, stride: int, device=None) -> torch.Tensor:
    """Boolean ``[N, N]`` selection of the Dozer self-attention (Eqs. 2-3).

    Local keeps ``|i - j| <= floor(w / 2)``; Stride keeps ``|i - j|`` that is a
    multiple of the interval ``stride + 1`` (the official interval convention).
    """
    idx = torch.arange(num_patches, device=device)
    offset = (idx.view(1, -1) - idx.view(-1, 1)).abs()
    mask = torch.zeros(num_patches, num_patches, dtype=torch.bool, device=device)
    if local_window:
        mask |= offset <= local_window // 2
    if stride:
        mask |= torch.remainder(offset, stride + 1) == 0
    return mask


def dozer_cross_mask(
    num_queries: int,
    num_keys: int,
    num_pred_queries: int,
    local_window: int,
    stride: int,
    vary_len: int,
    device=None,
) -> torch.Tensor:
    """Boolean ``[N_dec, N_enc]`` selection of the Dozer cross-attention (Eqs. 2-4).

    Local: every query sees the last ``floor(w/2)`` encoder patches (``1`` when
    ``w = 1``). Stride: query ``i`` sees keys ``j`` with
    ``j - i = N_enc - floor(N_dec / 2) + m (stride + 1)`` for integer ``m`` and
    ``j - i > -N_enc``. Vary: the first horizon query sees the last ``vary_len``
    encoder patches and every later query one patch more.
    """
    q = torch.arange(num_queries, device=device).view(-1, 1)
    k = torch.arange(num_keys, device=device).view(1, -1)
    mask = torch.zeros(num_queries, num_keys, dtype=torch.bool, device=device)
    if local_window:
        last = local_window // 2 if local_window > 1 else local_window
        mask |= (k >= num_keys - last).expand(num_queries, num_keys)
    if stride:
        offset = k - q
        anchor = num_keys - num_queries // 2
        mask |= (torch.remainder(offset - anchor, stride + 1) == 0) & (offset > -num_keys)
    # Vary: query i keeps the last (i - (N_dec - N_pred) + vary_len) encoder patches.
    first_horizon = num_queries - num_pred_queries
    keep = q - first_horizon + vary_len
    mask |= k >= num_keys - keep
    return mask


class DozerAttention(nn.Module):
    """Multi-head attention restricted to a Dozer selection of query-key pairs.

    ``score_mode="zero"`` reproduces the official scores: unselected pairs get a
    pre-softmax score of 0 (they stay in the softmax). ``"masked"`` removes them
    (score ``-inf``); a query with no selected key then attends uniformly.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float, score_mode: ScoreMode = "zero") -> None:
        super().__init__()
        self.n_heads = n_heads
        self.score_mode = score_mode
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        self.out_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def attention_weights(self, queries: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Return ``[B, H, L_q, L_k]`` weights for per-head ``[B, L, H, E]`` inputs."""
        scale = 1.0 / math.sqrt(queries.shape[-1])
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.score_mode == "zero":
            scores = scores * mask.to(scores.dtype)
        else:
            masked = scores.masked_fill(~mask, float("-inf"))
            scores = torch.where(mask.any(dim=-1, keepdim=True), masked, torch.zeros_like(scores))
        return torch.softmax(scale * scores, dim=-1)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch, length_q, _ = queries.shape
        length_k = keys.shape[1]
        heads = self.n_heads
        q = self.query_projection(queries).view(batch, length_q, heads, -1)
        k = self.key_projection(keys).view(batch, length_k, heads, -1)
        v = self.value_projection(values).view(batch, length_k, heads, -1)
        weights = self.dropout(self.attention_weights(q, k, mask))
        out = torch.einsum("bhls,bshd->blhd", weights, v).reshape(batch, length_q, -1)
        return self.out_projection(out)


class DepthwiseConvFeedForward(nn.Module):
    """Pointwise conv, GELU, depthwise conv (kernel 3 over patches), pointwise conv."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.depthwise = nn.Conv1d(d_ff, d_ff, kernel_size=3, padding=1, groups=d_ff)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dropout(F.gelu(self.conv1(x.transpose(1, 2))))
        y = self.depthwise(y)
        return self.dropout(self.conv2(y).transpose(1, 2))


class DozerEncoderLayer(nn.Module):
    """Post-norm encoder layer: Dozer self-attention then the depthwise-conv FFN."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float, score_mode: ScoreMode) -> None:
        super().__init__()
        self.attention = DozerAttention(d_model, n_heads, dropout, score_mode)
        self.feed_forward = DepthwiseConvFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x, x, x, mask)))
        return self.norm2(x + self.feed_forward(x))


class DozerDecoderLayer(nn.Module):
    """Post-norm decoder layer: Dozer self-attention, Dozer cross-attention, FFN."""

    def __init__(self, d_model: int, d_ff: int, n_heads: int, dropout: float, score_mode: ScoreMode) -> None:
        super().__init__()
        self.self_attention = DozerAttention(d_model, n_heads, dropout, score_mode)
        self.cross_attention = DozerAttention(d_model, n_heads, dropout, score_mode)
        self.feed_forward = DepthwiseConvFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, self_mask, cross_mask):
        x = self.norm1(x + self.dropout(self.self_attention(x, x, x, self_mask)))
        x = self.norm2(x + self.dropout(self.cross_attention(x, memory, memory, cross_mask)))
        return self.norm3(x + self.feed_forward(x))


class DimensionInvariantEmbedding(nn.Module):
    """DI embedding (Eq. 7): ``tanh`` of a ``3 x 1`` conv lifting ``[B, L, D]`` to ``[B, c, L, D]``."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, embed_dim, kernel_size=(3, 1), padding=(1, 0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.conv(x.unsqueeze(1)))


class Model(nn.Module):
    """Dozerformer forecaster with the four-input TSFLab interface."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        label_len: int = 96,
        patch_size: int = 24,
        embed_dim: int = 8,
        d_ff: int = 32,
        n_heads: int = 4,
        e_layers: int = 2,
        d_layers: int = 1,
        local_window: int = 3,
        stride: int = 7,
        vary_len: int = 1,
        moving_avg: tuple[int, ...] | list[int] = (13, 17),
        trend_branch: bool = True,
        dropout: float = 0.2,
        score_mode: ScoreMode = "zero",
        normalize_decoder_input: bool = False,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.enc_in = enc_in
        self.patch_size = patch_size
        self.trend_branch = trend_branch
        self.normalize_decoder_input = normalize_decoder_input
        self.enc_patches = math.ceil(seq_len / patch_size)
        self.dec_len = label_len + pred_len
        self.dec_patches = math.ceil(self.dec_len / patch_size)
        self.pred_patches = math.ceil(pred_len / patch_size)
        d_model = embed_dim * patch_size

        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        self.moving_averages = nn.ModuleList(EdgePaddedMovingAverage(k) for k in moving_avg) if trend_branch else nn.ModuleList()
        self.trend_linear = nn.Linear(seq_len, pred_len) if trend_branch else None

        self.enc_embedding = DimensionInvariantEmbedding(embed_dim)
        self.dec_embedding = DimensionInvariantEmbedding(embed_dim)
        self.enc_position = nn.Parameter(torch.randn(1, embed_dim, self.enc_patches, patch_size, enc_in))
        self.dec_position = nn.Parameter(torch.randn(1, embed_dim, self.dec_patches, patch_size, enc_in))
        self.enc_pre_norm = nn.LayerNorm(d_model)
        self.enc_post_norm = nn.LayerNorm(d_model)
        self.dec_pre_norm = nn.LayerNorm(d_model)
        self.dec_post_norm = nn.LayerNorm(d_model)
        ff = d_ff * patch_size
        self.encoder = nn.ModuleList(DozerEncoderLayer(d_model, ff, n_heads, dropout, score_mode) for _ in range(e_layers))
        self.decoder = nn.ModuleList(DozerDecoderLayer(d_model, ff, n_heads, dropout, score_mode) for _ in range(d_layers))
        self.output_conv = nn.Conv2d(embed_dim, 1, kernel_size=1)

        self.register_buffer("enc_self_mask", dozer_self_mask(self.enc_patches, local_window, stride), persistent=False)
        self.register_buffer("dec_self_mask", dozer_self_mask(self.dec_patches, local_window, stride), persistent=False)
        self.register_buffer(
            "cross_mask",
            dozer_cross_mask(self.dec_patches, self.enc_patches, self.pred_patches, local_window, stride, vary_len),
            persistent=False,
        )

    def decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Eq. (5): the mean of edge-padded moving averages is the trend, the rest is seasonal."""
        trend = torch.stack([avg(x) for avg in self.moving_averages], dim=0).mean(dim=0)
        return x - trend, trend

    def decoder_input(self, raw: torch.Tensor, normalized: torch.Tensor) -> torch.Tensor:
        """Last ``label_len`` history steps then ``pred_len`` zeros: ``[B, label_len + pred_len, D]``."""
        source = normalized if self.normalize_decoder_input else raw
        history = source[:, source.shape[1] - self.label_len :, :]
        zeros = source.new_zeros(source.shape[0], self.pred_len, source.shape[2])
        return torch.cat([history, zeros], dim=1)

    def patchify(self, x: torch.Tensor, num_patches: int) -> torch.Tensor:
        """Zero-pad ``[B, c, L, D]`` at the end of time and split into ``[B, c, N, p, D]``."""
        pad = num_patches * self.patch_size - x.shape[2]
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        batch, channels, _, variables = x.shape
        return x.reshape(batch, channels, num_patches, self.patch_size, variables)

    @staticmethod
    def to_tokens(x: torch.Tensor) -> torch.Tensor:
        """``[B, c, N, p, D] -> [B * D, N, p * c]`` (channel-independent patch tokens)."""
        batch, channels, patches, size, variables = x.shape
        return x.permute(0, 4, 2, 3, 1).reshape(batch * variables, patches, size * channels)

    @staticmethod
    def from_tokens(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """Inverse of :meth:`to_tokens` for a tensor shaped like ``like``."""
        batch, channels, patches, size, variables = like.shape
        return x.reshape(batch, variables, patches, size, channels).permute(0, 4, 2, 3, 1)

    def encode(self, seasonal: torch.Tensor) -> torch.Tensor:
        """Embed, patch, add positions, pre-norm, Dozer encoder, post-norm, add the embedding back."""
        identity = self.patchify(self.enc_embedding(seasonal), self.enc_patches)
        tokens = self.enc_pre_norm(self.to_tokens(identity + self.enc_position))
        for layer in self.encoder:
            tokens = layer(tokens, self.enc_self_mask)
        return self.from_tokens(self.enc_post_norm(tokens), identity) + identity

    def decode(self, seasonal_dec: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Same pipeline as :meth:`encode` with Dozer self- and cross-attention to ``memory``."""
        identity = self.patchify(self.dec_embedding(seasonal_dec), self.dec_patches)
        tokens = self.dec_pre_norm(self.to_tokens(identity + self.dec_position))
        cross = self.to_tokens(memory)
        for layer in self.decoder:
            tokens = layer(tokens, cross, self.dec_self_mask, self.cross_mask)
        return self.from_tokens(self.dec_post_norm(tokens), identity) + identity

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        normalized = self.revin(x_enc, "norm")
        dec_in = self.decoder_input(x_enc, normalized)
        if self.trend_branch:
            seasonal, trend = self.decompose(normalized)
            seasonal_dec, _ = self.decompose(dec_in)
        else:
            seasonal, trend, seasonal_dec = normalized, None, dec_in
        hidden = self.decode(seasonal_dec, self.encode(seasonal))
        batch, channels, patches, size, variables = hidden.shape
        hidden = hidden.reshape(batch, channels, patches * size, variables)[:, :, : self.dec_len]
        # Eq. (10): 1 x 1 conv from c feature maps to one seasonal forecast.
        forecast = self.output_conv(hidden).squeeze(1)[:, -self.pred_len :, :]
        if trend is not None:
            # Eqs. (6) and (11): linear trend forecast added to the seasonal one.
            forecast = forecast + self.trend_linear(trend.transpose(1, 2)).transpose(1, 2)
        return self.revin(forecast, "denorm")
