"""Canonical TSMixer-style time/feature residual mixing block.

Extracted verbatim (structure and attribute names unchanged) from the basic
TSMixer implementation (paper Appendix B.3.2): pre-normalized residual time
mixing over the sequence axis followed by a pre-normalized residual two-layer
feature (channel) mixer. Reuse this block only where a consumer's mixing
order, normalization shape, and residual placement are exactly this one;
paper-specific factorizations (subsequence mixers, low-rank channel
bottlenecks, post-norm ordering, additional hidden layers, etc.) must stay
model-local instead of being forced through this contract.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MixerBlock(nn.Module):
    """Paper time mixing followed by feature mixing, both residual."""

    def __init__(self, seq_len: int, channels: int, hidden: int, dropout: float) -> None:
        super().__init__()
        normalized_shape = (seq_len, channels)
        self.time_norm = nn.LayerNorm(normalized_shape)
        self.feature_norm = nn.LayerNorm(normalized_shape)
        self.time_projection = nn.Linear(seq_len, seq_len)
        self.feature_in = nn.Linear(channels, hidden)
        self.feature_out = nn.Linear(hidden, channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_input = self.time_norm(x).transpose(1, 2)
        time_delta = self.dropout(self.activation(self.time_projection(time_input)))
        x = x + time_delta.transpose(1, 2)
        feature_input = self.feature_norm(x)
        feature_delta = self.feature_out(self.dropout(self.activation(self.feature_in(feature_input))))
        return x + self.dropout(feature_delta)


__all__ = ["MixerBlock"]
