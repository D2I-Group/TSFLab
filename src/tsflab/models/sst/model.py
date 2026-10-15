"""SST: multi-scale hybrid Mamba-Transformer experts with a long-short router.

Independent implementation from Section 4 (Definition 4.1, Eq. 6, Figs. 5-6) of
Xu et al., "SST: Multi-Scale Hybrid Mamba-Transformer Experts for Time Series
Forecasting" (arXiv 2404.14757, CIKM 2025), after reading the pinned official code
(``XiongxiaoXu/SST`` at ``b39292bf``, no license file) to resolve omissions;
nothing is copied. The selective SSM is the cataloged pure-PyTorch ``mamba``
block instead of the ``mamba_ssm`` CUDA kernels.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.mamba import MambaBlock
from tsflab.models._components.revin import RevIN


def patch_count(length: int, patch_len: int, stride: int) -> int:
    """Number of patches after end-padding by ``stride`` copies of the last value:
    ``floor((length - patch_len) / stride) + 2``."""
    return (length - patch_len) // stride + 2


def patchify(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """Multi-scale patcher (Section 4.1) on ``[B, C, T]``: replicate the last value
    ``stride`` times, then cut ``[B, C, N, patch_len]`` patches with the given stride."""
    padded = F.pad(x, (0, stride), mode="replicate")
    return padded.unfold(-1, patch_len, stride)


def pts_resolution(patch_len: int, stride: int) -> float:
    """Definition 4.1 / Eq. (6), approximate PTS resolution ``sqrt(P) / Str``."""
    return math.sqrt(patch_len) / stride


def local_window_mask(tokens: int, window: int, device=None) -> torch.Tensor:
    """Fig. 6: boolean ``[tokens, tokens]`` mask, True where attention is blocked,
    i.e. outside the symmetric window ``|i - j| <= window // 2``."""
    index = torch.arange(tokens, device=device)
    return (index[:, None] - index[None, :]).abs() > window // 2


class PatternsExpertLayer(nn.Module):
    """One patterns-expert layer: a Mamba mixer followed by a position-wise
    feed-forward network (Fig. 5); the official layer has no residual or norm."""

    def __init__(self, d_model: int, d_ff: int, d_state: int, d_conv: int) -> None:
        super().__init__()
        self.mamba = MambaBlock(d_model, 2 * d_model, math.ceil(d_model / 16), d_conv, d_state, reference_dt_init=True)
        self.ff_in = nn.Linear(d_model, d_ff)
        self.ff_out = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff_out(F.gelu(self.ff_in(self.mamba(x))))


class LocalWindowAttention(nn.Module):
    """Multi-head scaled dot-product attention restricted to a local window, with the
    pre-softmax scores of the previous layer added (residual attention)."""

    def __init__(self, d_model: int, n_heads: int, window: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Sequential(nn.Linear(d_model, d_model), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor, prev: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, tokens, _ = x.shape

        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, tokens, self.n_heads, self.head_dim).transpose(1, 2)

        q, k, v = heads(self.query(x)), heads(self.key(x)), heads(self.value(x))
        scores = q @ k.transpose(-1, -2) * self.head_dim**-0.5
        if prev is not None:
            scores = scores + prev
        scores = scores.masked_fill(local_window_mask(tokens, self.window, x.device), float("-inf"))
        mixed = torch.softmax(scores, dim=-1) @ v
        return self.out(mixed.transpose(1, 2).reshape(batch, tokens, -1)), scores


class _BatchNormTokens(nn.Module):
    """BatchNorm1d over the feature axis of ``[B, N, D]`` tokens."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x.transpose(1, 2)).transpose(1, 2)


class VariationsExpertLayer(nn.Module):
    """One local window Transformer (LWT) layer: post-norm local attention and
    feed-forward sublayers with BatchNorm, as in the PatchTST encoder."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, window: int, dropout: float) -> None:
        super().__init__()
        self.attention = LocalWindowAttention(d_model, n_heads, window, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.attention_norm = _BatchNormTokens(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_norm = _BatchNormTokens(d_model)

    def forward(self, x: torch.Tensor, prev: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        attended, scores = self.attention(x, prev)
        x = self.attention_norm(x + self.attention_dropout(attended))
        x = self.ff_norm(x + self.ff_dropout(self.ff(x)))
        return x, scores


class LongShortRouter(nn.Module):
    """Section 4.3: project the (normalized) long-range input over variates to
    ``d_model``, flatten, and map to two softmax weights ``(p_L, p_S)`` per sample."""

    def __init__(self, seq_len: int, enc_in: int, d_model: int) -> None:
        super().__init__()
        self.project = nn.Linear(enc_in, d_model)
        self.weigh = nn.Linear(seq_len * d_model, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prob = torch.softmax(self.weigh(self.project(x).flatten(1)), dim=-1)
        return prob[:, 0], prob[:, 1]


class Model(nn.Module):
    """SST forecaster for ``[B, seq_len, enc_in]`` histories."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        short_len: int | None = None,
        m_patch_len: int = 48,
        m_stride: int = 16,
        m_layers: int = 1,
        d_state: int = 16,
        d_conv: int = 4,
        patch_len: int = 16,
        stride: int = 8,
        local_ws: int = 7,
        e_layers: int = 3,
        n_heads: int = 4,
        d_model: int = 16,
        d_ff: int = 128,
        dropout: float = 0.3,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        short_len = seq_len // 2 if short_len is None else short_len
        if not 0 < short_len <= seq_len:
            raise ValueError("short_len must lie in (0, seq_len]")
        if m_patch_len > seq_len or patch_len > short_len:
            raise ValueError("patch lengths must not exceed their range lengths")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.short_len = short_len
        self.m_patch_len, self.m_stride = m_patch_len, m_stride
        self.patch_len, self.stride = patch_len, stride
        self.long_patches = patch_count(seq_len, m_patch_len, m_stride)
        self.short_patches = patch_count(short_len, patch_len, stride)

        self.revin = RevIN(enc_in, affine=False)
        self.router = LongShortRouter(seq_len, enc_in, d_model)
        # Patterns expert: low-resolution long-range patches -> Mamba layers.
        self.long_embedding = nn.Linear(m_patch_len, d_model)
        self.patterns_expert = nn.ModuleList(
            PatternsExpertLayer(d_model, d_ff, d_state, d_conv) for _ in range(m_layers)
        )
        # Variations expert: high-resolution short-range patches -> LWT layers.
        self.short_embedding = nn.Linear(patch_len, d_model)
        self.position = nn.Parameter(torch.empty(self.short_patches, d_model).uniform_(-0.02, 0.02))
        self.embedding_dropout = nn.Dropout(dropout)
        self.variations_expert = nn.ModuleList(
            VariationsExpertLayer(d_model, n_heads, d_ff, local_ws, dropout) for _ in range(e_layers)
        )
        # Forecasting module: weighted concatenation, flatten, linear head.
        self.head = nn.Linear(d_model * (self.long_patches + self.short_patches), pred_len)
        self.head_dropout = nn.Dropout(head_dropout)

    def encode_long(self, x: torch.Tensor) -> torch.Tensor:
        """Patterns expert on ``[B, C, L]`` -> ``[B, C, N_L, D]``."""
        patches = patchify(x, self.m_patch_len, self.m_stride)
        batch, channels = patches.shape[:2]
        z = self.long_embedding(patches.flatten(0, 1))
        for layer in self.patterns_expert:
            z = layer(z)
        return z.reshape(batch, channels, self.long_patches, -1)

    def encode_short(self, x: torch.Tensor) -> torch.Tensor:
        """Variations expert on the latest ``short_len`` steps ``[B, C, S]`` -> ``[B, C, N_S, D]``."""
        patches = patchify(x, self.patch_len, self.stride)
        batch, channels = patches.shape[:2]
        z = self.embedding_dropout(self.short_embedding(patches.flatten(0, 1)) + self.position)
        scores = None
        for layer in self.variations_expert:
            z, scores = layer(z, scores)
        return z.reshape(batch, channels, self.short_patches, -1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = self.revin(x_enc, "norm")
        p_long, p_short = self.router(x)
        series = x.transpose(1, 2)  # [B, C, L]
        z_long = self.encode_long(series).flatten(2)
        z_short = self.encode_short(series[..., -self.short_len :]).flatten(2)
        fused = torch.cat((p_long.view(-1, 1, 1) * z_long, p_short.view(-1, 1, 1) * z_short), dim=-1)
        out = self.head_dropout(self.head(fused)).transpose(1, 2)
        return self.revin(out, "denorm")
