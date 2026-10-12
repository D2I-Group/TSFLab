"""PENGUIN (Sun, Chen & Sun, AISTATS 2026, arXiv:2508.13773).

Independent implementation of the encoder-only forecaster: RevIN, channel-
independent patch embedding, Transformer layers whose attention is a grouped
multi-query attention with a periodic-nested ALiBi bias (one key/value pair per
head group, one period per group), and a flatten-linear head.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.embed import PatchEmbedding
from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.mamba import RMSNorm
from tsflab.models._components.periodic_alibi_bias import periodic_alibi_bias
from tsflab.models._components.revin import RevIN


class PeriodicNestedGroupAttention(nn.Module):
    """Eq. 4: ``g`` groups of ``n = h / g`` query heads, each group sharing one K/V head.

    Scores are ``Q K^T / sqrt(d_h) + B`` with ``B`` the periodic-nested bias of the
    group (Eq. 5-6); an optional causal mask hides later patches.
    """

    def __init__(
        self, d_model: int, n_heads: int, periods: Sequence[int] | None, dropout: float,
        alibi: bool, causal: bool,
    ) -> None:
        super().__init__()
        self.periods = tuple(periods) if periods else None
        self.groups = len(self.periods) if self.periods else 1
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.alibi, self.causal = alibi, causal
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, self.groups * self.head_dim)
        self.value = nn.Linear(d_model, self.groups * self.head_dim)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        g, per = self.groups, self.n_heads // self.groups
        q = self.query(x).view(b, n, g, per, self.head_dim)
        k = self.key(x).view(b, n, g, self.head_dim)
        v = self.value(x).view(b, n, g, self.head_dim)
        scores = torch.einsum("blgpe,bsge->bgpls", q, k) / math.sqrt(self.head_dim)
        if self.alibi:
            bias = periodic_alibi_bias(n, self.n_heads, self.periods, device=x.device, dtype=x.dtype)
            scores = scores + bias.view(g, per, n, n)
        if self.causal:
            hidden = torch.ones(n, n, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(hidden, float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.einsum("bgpls,bsge->blgpe", weights, v)
        return self.out(context.reshape(b, n, -1))


class EncoderLayer(nn.Module):
    """Attention and feed-forward sub-layers, each followed by residual add and norm."""

    def __init__(
        self, d_model: int, d_ff: int, n_heads: int, periods: Sequence[int] | None, dropout: float,
        activation: str, use_rmsnorm: bool, alibi: bool, causal: bool,
    ) -> None:
        super().__init__()
        self.attention = PeriodicNestedGroupAttention(d_model, n_heads, periods, dropout, alibi, causal)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        norm = (lambda: RMSNorm(d_model, eps=1e-6)) if use_rmsnorm else (lambda: nn.LayerNorm(d_model))
        self.norm1, self.norm2 = norm(), norm()
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout(self.attention(x)))
        y = self.dropout(self.activation(self.fc1(x)))
        y = self.dropout(self.fc2(y))
        return self.norm2(x + y)


def _token_periods(periods: Sequence[int], stride: int) -> list[int] | None:
    periods = tuple(periods)
    return [p // stride for p in periods] or None  # P_S = P / S (Sec. 4.2.2)


class Model(nn.Module):
    """PENGUIN forecaster: [B, L, C] history -> [B, H, C] forecast."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        d_ff: int = 256,
        n_heads: int = 8,
        e_layers: int = 2,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        periods: Sequence[int] = (24,),
        activation: str = "relu",
        use_rmsnorm: bool = False,
        alibi: bool = True,
        causal: bool = True,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("invalid PENGUIN dimension")
        if seq_len < patch_len:
            raise ValueError("seq_len must be at least patch_len")
        token_periods = _token_periods(periods, stride)
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.num_patches = (seq_len - patch_len) // stride + 2  # one replicate-padded patch at the end

        self.revin = RevIN(enc_in, affine=False)
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, stride, dropout)
        self.layers = nn.ModuleList(
            EncoderLayer(d_model, d_ff, n_heads, token_periods, dropout, activation, use_rmsnorm, alibi, causal)
            for _ in range(e_layers)
        )
        self.final_norm = nn.BatchNorm1d(d_model)
        self.head = FlattenForecastHead(False, enc_in, d_model * self.num_patches, pred_len, dropout)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected (*,{self.seq_len},{self.enc_in}), got {tuple(x_enc.shape)}")
        x = self.revin(x_enc, "norm").transpose(1, 2)  # [B, C, L]
        tokens, n_vars = self.patch_embedding(x)  # [B*C, N, d]
        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.final_norm(tokens.transpose(1, 2)).transpose(1, 2)
        tokens = tokens.reshape(-1, n_vars, tokens.shape[1], tokens.shape[2])  # [B, C, N, d]
        forecast = self.head(tokens.permute(0, 1, 3, 2))  # [B, C, d, N] -> [B, C, H]
        return self.revin(forecast.transpose(1, 2), "denorm")
