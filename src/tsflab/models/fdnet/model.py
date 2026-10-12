"""Local FDNet implementation (paper Sections 5.1-5.2, Figs. 4-6).

Independent rewrite from the paper; the pinned official repository
(Apache-2.0) was read only to resolve omissions such as the sub-sequence
boundaries, the per-sub-sequence depth schedule, the dropout placement, and
the residual grouping of the four convolutions in a feature extraction layer.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm


def focal_lengths(seq_len: int, focal_parts: int) -> list[int]:
    """Sub-sequence lengths from the latest to the farthest (Section 5.2).

    With ``f`` parts the proportions are ``1/2^(f-1)`` for the two latest
    sub-sequences and ``1/2^(f-2), ..., 1/2`` for the farther ones, so the
    lengths sum to ``seq_len`` (e.g. ``f = 4``: 1/8, 1/8, 1/4, 1/2).
    """
    if focal_parts == 1:
        return [seq_len]
    shortest = seq_len // 2 ** (focal_parts - 1)
    return [shortest] + [shortest * 2**i for i in range(focal_parts - 1)]


def focal_depths(num_layers: int, focal_parts: int) -> list[int]:
    """Feature extraction depths from the latest to the farthest sub-sequence.

    Official schedule: the two latest sub-sequences use ``N`` layers and each
    farther one loses a layer (``N, N, N-1, ..., N-f+2``).
    """
    if focal_parts == 1:
        return [num_layers]
    return [num_layers] + [num_layers - i for i in range(focal_parts - 1)]


def focal_split(x: torch.Tensor, focal_parts: int) -> list[torch.Tensor]:
    """Split ``[batch, time, variates]`` into consecutive sub-sequences, latest first."""
    parts = []
    end = x.shape[1]
    for length in focal_lengths(x.shape[1], focal_parts):
        parts.append(x[:, end - length : end])
        end -= length
    return parts


class WeightNormConvUnit(nn.Module):
    """One residual pair of Fig. 5: WN 1x1 conv, GELU, WN kx1 conv, GELU, add input.

    Operates on ``[batch, d_model, time, variates]``; the variate kernel size is
    one, so variates never mix, and the temporal padding keeps the length.
    Dropout follows each convolution, before its activation.
    """

    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.pointwise = weight_norm(nn.Conv2d(d_model, d_model, kernel_size=(1, 1)))
        self.temporal = weight_norm(
            nn.Conv2d(
                d_model,
                d_model,
                kernel_size=(kernel_size, 1),
                padding=((kernel_size - 1) // 2, 0),
            )
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.activation(self.dropout(self.pointwise(x)))
        y = self.activation(self.dropout(self.temporal(y)))
        return x + y


class DecomposedFeatureExtractor(nn.Module):
    """Decomposed feature extraction layer (Fig. 5): two residual conv pairs, four convs."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.units = nn.Sequential(
            WeightNormConvUnit(d_model, kernel_size, dropout),
            WeightNormConvUnit(d_model, kernel_size, dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.units(x)


class DecomposedBranch(nn.Module):
    """FDNet without ICOM (Fig. 4) applied to one focal sub-sequence.

    Element-wise 1x1 embedding of every (time, variate) value to ``d_model``,
    ``depth`` decomposed feature extraction layers, then a per-variate flatten
    of ``[time, d_model]`` and one projection to the horizon shared by all
    variates (the paper's kernel-1 1D convolution).
    """

    def __init__(
        self,
        length: int,
        pred_len: int,
        d_model: int,
        kernel_size: int,
        depth: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.embedding = nn.Conv2d(1, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            DecomposedFeatureExtractor(d_model, kernel_size, dropout) for _ in range(depth)
        )
        self.projection = nn.Linear(length * d_model, pred_len)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, L, V]`` values to ``[B, D, L, V]`` local feature maps."""
        h = self.dropout(self.embedding(x.unsqueeze(1)))
        for layer in self.layers:
            h = layer(h)
        return h

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        # [B, D, L, V] -> [B, V, L, D] -> [B, V, L*D] -> [B, V, H] -> [B, H, V]
        flat = h.permute(0, 3, 2, 1).flatten(start_dim=2)
        return self.projection(flat).transpose(1, 2)


class Model(nn.Module):
    """FDNet: focal decomposition of the window into independent decomposed branches.

    Each focal sub-sequence gets its own branch whose depth decreases with its
    temporal distance to the horizon; the branch forecasts are averaged.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 8,
        kernel_size: int = 3,
        num_layers: int = 5,
        focal_parts: int = 5,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if seq_len % 2 ** (focal_parts - 1) != 0:
            raise ValueError(
                f"seq_len={seq_len} must be divisible by 2**(focal_parts - 1)={2 ** (focal_parts - 1)}"
            )
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.focal_parts = focal_parts
        lengths = focal_lengths(seq_len, focal_parts)
        depths = focal_depths(num_layers, focal_parts)
        self.branches = nn.ModuleList(
            DecomposedBranch(length, pred_len, d_model, kernel_size, depth, dropout)
            for length, depth in zip(lengths, depths)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.shape[1] != self.seq_len or x_enc.shape[2] != self.enc_in:
            raise ValueError(
                f"expected input [batch, {self.seq_len}, {self.enc_in}], got {tuple(x_enc.shape)}"
            )
        parts = focal_split(x_enc, self.focal_parts)
        outputs = [branch(part) for branch, part in zip(self.branches, parts)]
        return torch.stack(outputs, dim=0).mean(dim=0)
