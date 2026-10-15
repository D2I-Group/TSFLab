"""Compact Transformer encoder used by independent patch forecasters."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def _activation(value: str | Callable[[torch.Tensor], torch.Tensor]) -> str | Callable:
    if callable(value):
        return value
    normalized = value.lower()
    if normalized not in {"relu", "gelu"}:
        raise ValueError("activation must be relu, gelu, or a callable")
    return normalized


class TSTEncoder(nn.Module):
    """Stack batch-first self-attention blocks with an optional final norm."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        *,
        n_layers: int = 3,
        d_k: int | None = None,
        d_v: int | None = None,
        d_ff: int = 256,
        activation: str | Callable = "gelu",
        norm: str = "BatchNorm",
        attn_dropout: float = 0.0,
        res_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        pre_norm: bool = False,
        **_: object,
    ) -> None:
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        head_width = d_model // n_heads
        if d_k not in {None, head_width} or d_v not in {None, head_width}:
            raise ValueError("custom key/value widths are not supported by this encoder")
        dropout = max(attn_dropout, res_dropout, ffn_dropout, proj_dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=_activation(activation),
            batch_first=True,
            norm_first=pre_norm,
        )
        final_norm: nn.Module | None
        if "batch" in norm.lower():
            final_norm = _BatchFeatureNorm(d_model)
        elif "layer" in norm.lower():
            final_norm = nn.LayerNorm(d_model)
        else:
            raise ValueError("norm must select BatchNorm or LayerNorm")
        self.layers = nn.TransformerEncoder(layer, n_layers, norm=final_norm)

    def forward(self, values: torch.Tensor, **_: object) -> torch.Tensor:
        return self.layers(values)


class _BatchFeatureNorm(nn.Module):
    """Apply BatchNorm1d over features while retaining batch-first layout."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(width)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.norm(values.transpose(1, 2)).transpose(1, 2)
