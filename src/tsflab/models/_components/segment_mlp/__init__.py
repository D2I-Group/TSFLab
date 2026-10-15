"""Segment-to-token (or token-to-segment) map: one ``Linear`` or a plain MLP.

LLM-backbone forecasters that tokenize a series into non-overlapping segments
use this map in both directions: a segment of ``token_len`` values to one LLM
token (SegmentEmbedding) and an LLM hidden state back to a segment
(SegmentProjection). Moved verbatim from AutoTimes and TALON.
"""

from __future__ import annotations

import torch
import torch.nn as nn

ACTIVATIONS = {"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class SegmentMLP(nn.Module):
    """SegmentEmbedding / SegmentProjection: a linear map or a multi-layer perceptron.

    ``hidden_layers == 0`` is one ``Linear``; ``hidden_layers = n >= 2`` stacks
    ``n`` Linear layers with the activation and dropout between them.
    """

    def __init__(
        self, f_in: int, f_out: int, hidden_dim: int, hidden_layers: int, dropout: float, activation: str
    ) -> None:
        super().__init__()
        if hidden_layers == 0:
            self.layers = nn.Sequential(nn.Linear(f_in, f_out))
            return
        if hidden_layers < 2:
            raise ValueError("mlp_hidden_layers must be 0 (linear) or at least 2")
        width = [f_in] + [hidden_dim] * (hidden_layers - 1)
        layers: list[nn.Module] = []
        for a, b in zip(width[:-1], width[1:]):
            layers += [nn.Linear(a, b), ACTIVATIONS[activation](), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden_dim, f_out))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


__all__ = ["ACTIVATIONS", "SegmentMLP"]
