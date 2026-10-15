"""Independent, paper-derived Pyraformer forecasting model.

The ICLR 2022 method defines a coarser-scale construction module and a
pyramidal attention graph. This implementation was written locally from those
public equations; reference repository source code was not used.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._inputs import check_marks, check_window


def _raw_calendar_features(marks: torch.Tensor) -> torch.Tensor:
    """Normalize ``[year, month, day, weekday, hour, minute]`` marks."""
    scales = marks.new_tensor((50.0, 11.0, 30.0, 6.0, 23.0, 59.0))
    offsets = marks.new_tensor((2000.0, 1.0, 1.0, 0.0, 0.0, 0.0))
    return (marks - offsets) / scales - 0.5


def pyramid_sizes(length: int, branching: tuple[int, ...]) -> tuple[int, ...]:
    """Return exact scale sizes for a divisible C-ary temporal pyramid."""
    sizes = [length]
    for factor in branching:
        if sizes[-1] % factor:
            raise ValueError(f"scale length {sizes[-1]} must be divisible by branching factor {factor}")
        sizes.append(sizes[-1] // factor)
    return tuple(sizes)


def pyramid_neighbour_table(
    sizes: tuple[int, ...],
    branching: tuple[int, ...],
    neighbourhood_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build Equation-2 PAM neighbourhoods as padded node-index tables."""
    starts = [sum(sizes[:scale]) for scale in range(len(sizes))]
    radius = neighbourhood_size // 2
    rows: list[list[int]] = []
    for scale, size in enumerate(sizes):
        for local in range(size):
            rows.append(_pam_neighbours(sizes, branching, starts, radius, scale, local))
    return _padded_table(rows)


def _pam_neighbours(
    sizes: tuple[int, ...],
    branching: tuple[int, ...],
    starts: list[int],
    radius: int,
    scale: int,
    local: int,
) -> list[int]:
    """Sorted PAM neighbourhood of node ``local`` at ``scale``: siblings, children, parent."""
    size = sizes[scale]
    neighbours = {starts[scale] + index for index in range(max(0, local - radius), min(size, local + radius + 1))}
    if scale > 0:
        factor = branching[scale - 1]
        child_start = local * factor
        neighbours.update(starts[scale - 1] + child for child in range(child_start, child_start + factor))
    if scale + 1 < len(sizes):
        neighbours.add(starts[scale + 1] + local // branching[scale])
    return sorted(neighbours)


def _padded_table(rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad neighbour lists into an index table and its validity mask."""
    width = max(map(len, rows))
    indices = torch.zeros(len(rows), width, dtype=torch.long)
    valid = torch.zeros(len(rows), width, dtype=torch.bool)
    for row, neighbours in enumerate(rows):
        indices[row, : len(neighbours)] = torch.tensor(neighbours)
        valid[row, : len(neighbours)] = True
    return indices, valid


def finest_ancestor_table(sizes: tuple[int, ...], branching: tuple[int, ...]) -> torch.Tensor:
    """Map every finest-scale position to its node at every pyramid scale."""
    starts = [sum(sizes[:scale]) for scale in range(len(sizes))]
    rows = []
    for position in range(sizes[0]):
        ancestors = [position]
        local = position
        for scale, factor in enumerate(branching, start=1):
            local //= factor
            ancestors.append(starts[scale] + local)
        rows.append(ancestors)
    return torch.tensor(rows, dtype=torch.long)


class CoarseScaleConstructor(nn.Module):
    """Construct learned summaries at successively coarser temporal scales."""

    def __init__(self, d_model: int, branching: tuple[int, ...]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            nn.Conv1d(d_model, d_model, kernel_size=factor, stride=factor) for factor in branching
        )
        self.norms = nn.ModuleList(nn.LayerNorm(d_model) for _ in branching)

    def forward(self, finest: torch.Tensor) -> torch.Tensor:
        scales = [finest]
        current = finest
        for convolution, norm in zip(self.layers, self.norms, strict=True):
            current = convolution(current.transpose(1, 2)).transpose(1, 2)
            current = norm(F.gelu(current))
            scales.append(current)
        return torch.cat(scales, dim=1)


class PyramidalAttention(nn.Module):
    """Sparse multi-head attention over the paper-defined PAM graph."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        neighbours: torch.Tensor,
        neighbour_valid: torch.Tensor,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("neighbours", neighbours, persistent=True)
        self.register_buffer("neighbour_valid", neighbour_valid, persistent=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, nodes, width = x.shape
        shape = (batch, nodes, self.n_heads, self.head_dim)
        query = self.query(x).view(shape).transpose(1, 2)
        key = self.key(x).view(shape).transpose(1, 2)
        value = self.value(x).view(shape).transpose(1, 2)
        local_key = key[:, :, self.neighbours, :]
        local_value = value[:, :, self.neighbours, :]
        scores = torch.sum(query.unsqueeze(-2) * local_key, dim=-1)
        scores = scores / math.sqrt(self.head_dim)
        scores = scores.masked_fill(
            ~self.neighbour_valid.unsqueeze(0).unsqueeze(0),
            torch.finfo(scores.dtype).min,
        )
        weights = self.dropout(torch.softmax(scores, dim=-1))
        mixed = torch.sum(weights.unsqueeze(-1) * local_value, dim=-2)
        return self.output(mixed.transpose(1, 2).reshape(batch, nodes, width))


class PyramidalAttentionBlock(nn.Module):
    """Pre-normalized PAM attention and position-wise feed-forward block."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_heads: int,
        neighbours: torch.Tensor,
        neighbour_valid: torch.Tensor,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = PyramidalAttention(d_model, n_heads, neighbours, neighbour_valid, dropout)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        return x + self.dropout(self.feed_forward(self.feed_forward_norm(x)))


class Model(nn.Module):
    """Pyraformer clean-room rewrite for direct multi-horizon forecasting."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        window_size: tuple[int, ...] | list[int] = (4, 4),
        inner_size: int = 5,
    ) -> None:
        super().__init__()
        branching = tuple(window_size)
        sizes = pyramid_sizes(seq_len, branching)
        neighbours, neighbour_valid = pyramid_neighbour_table(sizes, branching, inner_size)
        ancestors = finest_ancestor_table(sizes, branching)

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.scale_count = len(sizes)
        self.value_embedding = nn.Linear(enc_in, d_model)
        self.calendar_embedding = nn.Linear(6, d_model, bias=False)
        self.register_buffer(
            "position_encoding",
            self._sinusoidal_position(seq_len, d_model),
            persistent=True,
        )
        self.coarse_scales = CoarseScaleConstructor(d_model, branching)
        self.blocks = nn.ModuleList(
            PyramidalAttentionBlock(
                d_model,
                d_ff,
                n_heads,
                neighbours,
                neighbour_valid,
                dropout,
            )
            for _ in range(e_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.register_buffer("ancestor_indices", ancestors, persistent=True)
        self.forecast_head = nn.Linear(self.scale_count * d_model, pred_len * enc_in)

    @staticmethod
    def _sinusoidal_position(length: int, width: int) -> torch.Tensor:
        position = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        exponent = torch.arange(0, width, 2, dtype=torch.float32)
        exponent = torch.exp(-math.log(10_000.0) * exponent / width)
        encoding = torch.zeros(length, width)
        encoding[:, 0::2] = torch.sin(position * exponent)
        if width > 1:
            encoding[:, 1::2] = torch.cos(position * exponent[: width // 2])
        return encoding.unsqueeze(0)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        check_marks(x_mark_enc, x_enc)
        if x_mark_enc is None:
            calendar = x_enc.new_zeros((*x_enc.shape[:2], 6))
        else:
            calendar = _raw_calendar_features(x_mark_enc.to(dtype=x_enc.dtype))

        finest = (
            self.value_embedding(x_enc)
            + self.calendar_embedding(calendar)
            + self.position_encoding.to(dtype=x_enc.dtype)
        )
        pyramid = self.coarse_scales(finest)
        for block in self.blocks:
            pyramid = block(pyramid)
        pyramid = self.final_norm(pyramid)

        # Prediction strategy 1: concatenate the last observed position and
        # its parent at every coarser scale, then forecast the full horizon.
        last_chain = pyramid[:, self.ancestor_indices[-1], :].flatten(1)
        return self.forecast_head(last_chain).view(x_enc.shape[0], self.pred_len, self.enc_in)
