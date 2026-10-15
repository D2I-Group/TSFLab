"""Autoformer-style progressive-decomposition encoder and decoder layers.

The token mixers (auto-correlation, Fourier blocks, attention, ...) are injected;
each is registered under a caller-chosen attribute name so consumers keep their
state-dict keys.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.series_decomposition import SeriesDecomposition


def decomposition_feed_forward(d_model: int, d_ff: int, dropout: float, activation: str) -> nn.Sequential:
    """Position-wise ``Linear -> GELU|ReLU -> Dropout -> Linear -> Dropout``."""
    nonlinearity: nn.Module = nn.GELU() if activation == "gelu" else nn.ReLU()
    return nn.Sequential(
        nn.Linear(d_model, d_ff),
        nonlinearity,
        nn.Dropout(dropout),
        nn.Linear(d_ff, d_model),
        nn.Dropout(dropout),
    )


class DecompositionEncoderLayer(nn.Module):
    """``s = decomp(x + mixer(x))[0]``; ``s = decomp(s + FFN(s))[0]``; trends are discarded."""

    def __init__(
        self,
        mixer: nn.Module,
        d_model: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
        activation: str,
        *,
        mixer_name: str = "mixer",
    ) -> None:
        super().__init__()
        self.mixer_name = mixer_name
        self.add_module(mixer_name, mixer)
        self.decomposition_one = SeriesDecomposition(moving_avg)
        self.feed_forward = decomposition_feed_forward(d_model, d_ff, dropout, activation)
        self.decomposition_two = SeriesDecomposition(moving_avg)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        mixer = self._modules[self.mixer_name]
        seasonal, _ = self.decomposition_one(values + mixer(values))
        seasonal, _ = self.decomposition_two(seasonal + self.feed_forward(seasonal))
        return seasonal


class DecompositionDecoderLayer(nn.Module):
    """Self mixer, cross mixer and FFN, each followed by a decomposition; the three
    extracted trends are projected to ``c_out`` (bias-free) and added to ``trend``."""

    def __init__(
        self,
        self_mixer: nn.Module,
        cross_mixer: nn.Module,
        d_model: int,
        d_ff: int,
        moving_avg: int,
        dropout: float,
        activation: str,
        c_out: int,
        *,
        mixer_names: tuple[str, str] = ("self_mixer", "cross_mixer"),
    ) -> None:
        super().__init__()
        if len(mixer_names) != 2 or mixer_names[0] == mixer_names[1]:
            raise ValueError("mixer_names must be two distinct attribute names")
        self.mixer_names = tuple(mixer_names)
        self.add_module(mixer_names[0], self_mixer)
        self.add_module(mixer_names[1], cross_mixer)
        self.feed_forward = decomposition_feed_forward(d_model, d_ff, dropout, activation)
        self.decompositions = nn.ModuleList([SeriesDecomposition(moving_avg) for _ in range(3)])
        self.trend_projections = nn.ModuleList([nn.Linear(d_model, c_out, bias=False) for _ in range(3)])

    def forward(
        self, seasonal: torch.Tensor, memory: torch.Tensor, trend: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self_mixer = self._modules[self.mixer_names[0]]
        cross_mixer = self._modules[self.mixer_names[1]]
        seasonal, trend_one = self.decompositions[0](seasonal + self_mixer(seasonal))
        seasonal, trend_two = self.decompositions[1](seasonal + cross_mixer(seasonal, memory))
        seasonal, trend_three = self.decompositions[2](seasonal + self.feed_forward(seasonal))
        for projection, extracted in zip(self.trend_projections, (trend_one, trend_two, trend_three)):
            trend = trend + projection(extracted)
        return seasonal, trend


__all__ = ["DecompositionDecoderLayer", "DecompositionEncoderLayer", "decomposition_feed_forward"]
