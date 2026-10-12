"""Position-table construction for patch-based sequence encoders."""

from __future__ import annotations

import math

import torch
from torch import nn


def _standardize(table: torch.Tensor) -> torch.Tensor:
    deviation = table.std(unbiased=True)
    if float(deviation) == 0.0:
        return table - table.mean()
    return (table - table.mean()) / (10.0 * deviation)


def _sinusoidal(length: int, width: int) -> torch.Tensor:
    positions = torch.arange(length, dtype=torch.float32)[:, None]
    pairs = torch.arange(0, width, 2, dtype=torch.float32)
    angular = positions * torch.exp(-math.log(10_000.0) * pairs / width)[None, :]
    table = torch.empty(length, width)
    table[:, 0::2] = angular.sin()
    if width > 1:
        table[:, 1::2] = angular[:, : table[:, 1::2].shape[1]].cos()
    return _standardize(table)


def _coordinate(length: int, width: int, power: float) -> torch.Tensor:
    time = torch.linspace(0.0, 1.0, length)[:, None].pow(power)
    if width == 1:
        return _standardize(2.0 * time - 1.0)
    feature = torch.linspace(0.0, 1.0, width)[None, :].pow(power)
    return _standardize(2.0 * time * feature - 1.0)


def _column(length: int, width: int, low: float, high: float) -> torch.Tensor:
    return torch.empty(length, 1).uniform_(low, high)


def _normal_column(length: int, width: int) -> torch.Tensor:
    return torch.empty(length, 1).normal_(mean=0.0, std=0.1)


def _small_uniform_table(length: int, width: int) -> torch.Tensor:
    return torch.empty(length, width).uniform_(-0.02, 0.02)


_POSITIONAL_TABLES = {
    "zero": lambda length, width: _column(length, width, -0.02, 0.02),
    "small_uniform": lambda length, width: _column(length, width, -0.02, 0.02),
    "normal": _normal_column,
    "gauss": _normal_column,
    "uniform": lambda length, width: _column(length, width, 0.0, 0.1),
    "zeros": _small_uniform_table,
    "small_uniform_2d": _small_uniform_table,
    "sincos": _sinusoidal,
    "lin1d": lambda length, width: _coordinate(length, 1, 1.0),
    "exp1d": lambda length, width: _coordinate(length, 1, 0.5),
    "lin2d": lambda length, width: _coordinate(length, width, 1.0),
    "exp2d": lambda length, width: _coordinate(length, width, 0.5),
}


def positional_encoding(
    kind: str | None,
    learnable: bool,
    length: int,
    width: int,
) -> nn.Parameter:
    """Create a positional table using the repository's stable public modes.

    Naming caveat: ``kind="zero"`` is NOT all zeros. It is a width-1 column of
    small uniform noise in ``[-0.02, 0.02]`` broadcast over features, and
    ``"zeros"`` is a full ``[length, width]`` table of the same noise. The
    legacy strings are kept so existing configs and checkpoints keep working;
    ``"small_uniform"`` is accepted as a clearer alias of ``"zero"`` and
    ``"small_uniform_2d"`` of ``"zeros"``.
    """
    if length < 1 or width < 1:
        raise ValueError("length and width must be positive")
    if kind is None:
        table = _small_uniform_table(length, width)
        learnable = False
    else:
        builder = _POSITIONAL_TABLES.get(kind)
        if builder is None:
            raise ValueError(f"unsupported positional encoding: {kind!r}")
        table = builder(length, width)
    return nn.Parameter(table, requires_grad=bool(learnable))
