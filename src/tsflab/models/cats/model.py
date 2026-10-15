"""Independent CATS implementation from the paper architecture."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CrossAttentionLayer(nn.Module):
    """Cross-attention only: future queries attend to historical patches."""

    def __init__(
        self,
        width: int,
        heads: int,
        hidden: int,
        dropout: float,
        attention_dropout: float,
        masking_probability: float,
        store_attention: bool,
    ) -> None:
        super().__init__()
        self.masking_probability = masking_probability
        self.store_attention = store_attention
        self.attention = nn.MultiheadAttention(width, heads, attention_dropout, batch_first=True)
        self.norm_attention = nn.LayerNorm(width)
        self.norm_feed_forward = nn.LayerNorm(width)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, width),
        )
        self.last_attention: torch.Tensor | None = None

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, weights = self.attention(queries, memory, memory, need_weights=self.store_attention)
        if self.training and self.masking_probability > 0:
            keep = (
                torch.rand(attended.shape[0], attended.shape[1], 1, device=attended.device) >= self.masking_probability
            )
            attended = attended * keep
        queries = self.norm_attention(queries + self.dropout(attended))
        queries = self.norm_feed_forward(queries + self.dropout(self.feed_forward(queries)))
        self.last_attention = weights if self.store_attention else None
        return queries


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int = 24,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        stride: int = 24,
        attn_dropout: float = 0.0,
        query_independence: bool = False,
        padding_patch: str | None = None,
        store_attn: bool = False,
        QAM_start: float = 0.1,
        QAM_end: float = 0.5,
    ) -> None:
        super().__init__()
        del seq_len, padding_patch
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.patch_len = patch_len
        self.stride = stride
        self.query_independence = query_independence
        self.output_patches = math.ceil(pred_len / patch_len)
        self.embedding = nn.Linear(patch_len, d_model)
        query_channels = enc_in if query_independence else 1
        self.future_queries = nn.Parameter(torch.empty(query_channels, self.output_patches, patch_len))
        nn.init.normal_(self.future_queries, std=patch_len**-0.5)
        probabilities = torch.linspace(QAM_start, QAM_end, max(n_layers, 1)).tolist()
        self.layers = nn.ModuleList(
            [
                CrossAttentionLayer(d_model, n_heads, d_ff, dropout, attn_dropout, probabilities[index], store_attn)
                for index in range(n_layers)
            ]
        )
        self.projection = nn.Linear(d_model, patch_len)

    def _patch(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] < self.patch_len:
            values = torch.nn.functional.pad(values, (self.patch_len - values.shape[-1], 0), mode="replicate")
        remainder = (values.shape[-1] - self.patch_len) % self.stride
        if remainder:
            values = torch.nn.functional.pad(values, (0, self.stride - remainder), mode="replicate")
        return values.unfold(-1, self.patch_len, self.stride)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        batch, _, channels = x_enc.shape
        centered = x_enc - x_enc[:, -1:, :]
        patches = self._patch(centered.transpose(1, 2))
        memory = self.embedding(patches).reshape(batch * channels, patches.shape[2], -1)
        raw_queries = (
            self.future_queries.expand(channels, -1, -1) if not self.query_independence else self.future_queries
        )
        queries = self.embedding(raw_queries).unsqueeze(0).expand(batch, -1, -1, -1)
        queries = queries.reshape(batch * channels, self.output_patches, -1)
        for layer in self.layers:
            queries = layer(queries, memory)
        forecast = self.projection(queries).reshape(batch, channels, -1)
        forecast = forecast[:, :, : self.pred_len].transpose(1, 2)
        return forecast + x_enc[:, -1:, :]
