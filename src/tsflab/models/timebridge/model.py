"""Independent TimeBridge implementation based on the paper equations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN


class AttentionBlock(nn.Module):
    def __init__(
        self, width: int, heads: int, hidden: int, attention_dropout: float, dropout: float, activation: str
    ) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, attention_dropout, batch_first=True)
        self.norm_attention = nn.LayerNorm(width)
        self.norm_feed_forward = nn.LayerNorm(width)
        nonlinearity: nn.Module = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.feed_forward = nn.Sequential(
            nn.Linear(width, hidden),
            nonlinearity,
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        attended, _ = self.attention(query, key, value, need_weights=False)
        result = self.norm_attention(query + self.dropout(attended))
        return self.norm_feed_forward(result + self.dropout(self.feed_forward(result)))


class IntegratedAttention(nn.Module):
    """Stationary patch scores (Q,K) applied to original non-stationary values."""

    def __init__(
        self,
        width: int,
        heads: int,
        hidden: int,
        stable_len: int,
        attention_dropout: float,
        dropout: float,
        activation: str,
    ) -> None:
        super().__init__()
        self.stable_len = stable_len
        self.block = AttentionBlock(width, heads, hidden, attention_dropout, dropout, activation)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        batch, channels, count, width = patches.shape
        sequence = patches.reshape(batch * channels, count, width)
        trend = F.avg_pool1d(
            F.pad(
                sequence.transpose(1, 2),
                (self.stable_len // 2, self.stable_len - 1 - self.stable_len // 2),
                mode="replicate",
            ),
            self.stable_len,
            stride=1,
        ).transpose(1, 2)
        stationary = sequence - trend
        result = self.block(stationary, stationary, sequence)
        return result.reshape(batch, channels, count, width)


class PatchDownsample(nn.Module):
    def __init__(
        self, width: int, heads: int, hidden: int, attention_dropout: float, dropout: float, activation: str
    ) -> None:
        super().__init__()
        self.block = AttentionBlock(width, heads, hidden, attention_dropout, dropout, activation)

    def forward(self, patches: torch.Tensor, target_count: int) -> torch.Tensor:
        batch, channels, count, width = patches.shape
        sequence = patches.reshape(batch * channels, count, width)
        queries = F.adaptive_avg_pool1d(sequence.transpose(1, 2), target_count).transpose(1, 2)
        result = self.block(queries, sequence, sequence)
        return result.reshape(batch, channels, target_count, width)


class CointegratedAttention(nn.Module):
    """Full attention across variates for each long-term patch aggregate."""

    def __init__(
        self, width: int, heads: int, hidden: int, attention_dropout: float, dropout: float, activation: str
    ) -> None:
        super().__init__()
        self.block = AttentionBlock(width, heads, hidden, attention_dropout, dropout, activation)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        batch, channels, count, width = patches.shape
        by_time = patches.permute(0, 2, 1, 3).reshape(batch * count, channels, width)
        result = self.block(by_time, by_time, by_time)
        return result.reshape(batch, count, channels, width).permute(0, 2, 1, 3)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        period: int = 24,
        num_p: int | None = None,
        ia_layers: int = 2,
        pd_layers: int = 1,
        ca_layers: int = 2,
        stable_len: int = 3,
        d_model: int = 16,
        n_heads: int = 4,
        d_ff: int = 128,
        attn_dropout: float = 0.15,
        dropout: float = 0.0,
        activation: str = "gelu",
        revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.period = period
        self.patch_count = num_p or math.ceil(seq_len / period)
        self.long_count = max(1, self.patch_count // max(stable_len, 1))
        self.normalization = RevIN(enc_in, enabled=revin)
        self.embedding = nn.Linear(period, d_model)
        self.integrated = nn.ModuleList(
            [
                IntegratedAttention(d_model, n_heads, d_ff, stable_len, attn_dropout, dropout, activation)
                for _ in range(ia_layers)
            ]
        )
        self.downsample = nn.ModuleList(
            [PatchDownsample(d_model, n_heads, d_ff, attn_dropout, dropout, activation) for _ in range(pd_layers)]
        )
        self.cointegrated = nn.ModuleList(
            [CointegratedAttention(d_model, n_heads, d_ff, attn_dropout, dropout, activation) for _ in range(ca_layers)]
        )
        self.projection = nn.Linear(self.long_count * d_model, pred_len)

    def _patch(self, values: torch.Tensor) -> torch.Tensor:
        target = self.patch_count * self.period
        if values.shape[1] < target:
            values = F.pad(values.transpose(1, 2), (target - values.shape[1], 0), mode="replicate").transpose(1, 2)
        else:
            values = values[:, -target:]
        batch, _, channels = values.shape
        return values.transpose(1, 2).reshape(batch, channels, self.patch_count, self.period)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        normalized = self.normalization(x_enc, "norm")
        patches = self.embedding(self._patch(normalized))
        for layer in self.integrated:
            patches = layer(patches)
        for layer in self.downsample:
            patches = layer(patches, self.long_count)
        if not self.downsample:
            patches = (
                F.adaptive_avg_pool1d(
                    patches.permute(0, 1, 3, 2).reshape(-1, patches.shape[-1], patches.shape[2]),
                    self.long_count,
                )
                .reshape(x_enc.shape[0], x_enc.shape[2], patches.shape[-1], self.long_count)
                .permute(0, 1, 3, 2)
            )
        for layer in self.cointegrated:
            patches = layer(patches)
        prediction = self.projection(patches.flatten(2)).transpose(1, 2)
        return self.normalization(prediction, "denorm")
