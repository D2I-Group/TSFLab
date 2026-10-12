"""Clean-room TimeExpert (arXiv:2509.23145): a channel-independent patch
Transformer whose vanilla self-attention is replaced by the cataloged
top-k local expert attention.
"""
from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.embed import PatchEmbedding
from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._components.topk_expert_attention import TopKExpertAttention


class TMOEBlock(nn.Module):
    """Top-k expert attention followed by a position-wise feed-forward network."""

    def __init__(self, d_model: int, n_heads: int, topk: int, shared: bool, dropout: float) -> None:
        super().__init__()
        self.attention = TopKExpertAttention(d_model, num_heads=n_heads, topk=topk, shared=shared, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(x)
        return x + self.feed_forward(x)


class Model(nn.Module):
    """TimeExpert: patches each channel independently, encodes them with
    stacked Temporal-Mix-of-Experts (TMOE) blocks -- self-attention where
    every key/value patch position is a candidate "expert" and each query
    routes to only its top-k most relevant experts, optionally alongside one
    shared global expert -- and forecasts with a shared flatten-and-linear
    head.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        patch_len: int = 16,
        stride: int = 8,
        dropout: float = 0.1,
        topk: int = 4,
        shared: bool = False,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len, enc_in, d_model, n_heads, e_layers, patch_len, stride) < 1:
            raise ValueError("invalid TimeExpert dimension")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in

        padding = stride
        self.patch_embedding = PatchEmbedding(d_model, patch_len, stride, padding, dropout)
        self.layers = nn.ModuleList(
            TMOEBlock(d_model, n_heads, topk, shared, dropout) for _ in range(e_layers)
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        num_patches = (seq_len - patch_len) // stride + 2
        self.head = FlattenForecastHead(
            individual=False, n_vars=enc_in, nf=d_model * num_patches, target_window=pred_len, head_dropout=dropout
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected (*,{self.seq_len},{self.enc_in})")

        normalized, mean, std = standardize(x_enc)

        channel_first = normalized.permute(0, 2, 1)
        tokens, n_vars = self.patch_embedding(channel_first)

        for layer in self.layers:
            tokens = layer(tokens)
        tokens = self.encoder_norm(tokens)

        tokens = tokens.reshape(-1, n_vars, tokens.shape[-2], tokens.shape[-1]).permute(0, 1, 3, 2)
        forecast = self.head(tokens).permute(0, 2, 1)

        return destandardize(forecast, mean, std)
