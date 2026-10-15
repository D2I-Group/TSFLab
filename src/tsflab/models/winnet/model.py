"""WinNet: one-convolution forecaster over a 2D sub-window tensor.

Independent implementation from Section 3 (Eqs. 1-5) of Ou et al., "WinNet: Make
Only One Convolutional Layer Effective for Time Series Forecasting" (arXiv
2311.00214), after reading the pinned official code (``ouwen18/WinNet`` at
``2012c30d``, no license file) to resolve omissions; nothing is copied.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.last_value_center import center_on_last_value, restore_last_value


def trend_pad(x: torch.Tensor, before: int, after: int) -> torch.Tensor:
    """Trend padding (Fig. 3) of a ``[..., R, C]`` tensor whose rows are consecutive
    segments of one sequence.

    Each row is extended on the left by the ``before`` points that precede it in the
    flattened sequence and on the right by the ``after`` points that follow it
    (zeros beyond the sequence ends); ``before`` and ``after`` zero rows are added
    above and below. The result is ``[..., R + before + after, C + before + after]``.
    """
    cols = x.shape[-1]
    flat = F.pad(x.flatten(-2), (before, after))
    widened = flat.unfold(-1, cols + before + after, cols)
    return F.pad(widened, (0, 0, before, after))


def decompose_2d(x: torch.Tensor, kernel: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-dimensional hybrid decomposition (Eq. 3) of ``[B, V, R, C]``.

    Trend: a ``kernel x kernel`` stride-1 average over the trend-padded tensor
    (``kernel // 2`` before, ``kernel - 1 - kernel // 2`` after, so the shape is
    kept); seasonal: ``x - trend``. Returns ``(trend, seasonal)``.
    """
    before = kernel // 2
    padded = trend_pad(x, before, kernel - 1 - before)
    batch, variables = padded.shape[:2]
    trend = F.avg_pool2d(padded.flatten(0, 1).unsqueeze(1), kernel, stride=1)
    trend = trend.view(batch, variables, *x.shape[-2:])
    return trend, x - trend


class DecompositionCorrelationBlock(nn.Module):
    """Eq. (4): per variable (CI), stack trend and seasonal as two input channels,
    trend-pad by one, apply one 3x3 Conv2d with a single output channel, then the
    sigmoid; per-variable outputs are concatenated again (CA). One independent
    kernel per variable, realised as a grouped convolution."""

    def __init__(self, enc_in: int) -> None:
        super().__init__()
        self.enc_in = enc_in
        self.conv = nn.Conv2d(2 * enc_in, enc_in, kernel_size=3, groups=enc_in)

    def forward(self, trend: torch.Tensor, seasonal: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack((trend, seasonal), dim=2)  # [B, V, 2, R, C]
        padded = trend_pad(stacked, 1, 1).flatten(1, 2)  # [B, 2V, R + 2, C + 2]
        return torch.sigmoid(self.conv(padded))


class Model(nn.Module):
    """WinNet forecaster for ``[B, seq_len, enc_in]`` histories."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        window: int = 24,
        dropout: float = 0.5,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.window = window
        # Eq. (1): map the history to window * window points before division.
        self.input_projection = nn.Linear(seq_len, window * window)
        self.dcb = DecompositionCorrelationBlock(enc_in)
        self.dropout = nn.Dropout(dropout)
        # Eq. (5): flatten the 2D tensor and map it to the horizon.
        self.output_projection = nn.Linear(window * window, pred_len)

    def sub_windows(self, x: torch.Tensor) -> torch.Tensor:
        """Eq. (1): linear map over time, then divide into ``window`` sub-windows of
        ``window`` consecutive points: ``[B, L, V] -> [B, V, window, window]``."""
        projected = self.input_projection(x.transpose(1, 2))
        return projected.view(x.shape[0], self.enc_in, self.window, self.window)

    def head(self, tensor: torch.Tensor) -> torch.Tensor:
        """One forecasting head: 2D decomposition (Eq. 3) then the DCB (Eq. 4)."""
        trend, seasonal = decompose_2d(tensor, self.window)
        return self.dropout(self.dcb(trend, seasonal))

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x, level = center_on_last_value(x_enc)
        within = self.sub_windows(x)
        # Eq. (2): the cross-window tensor is the transpose of the within-window one.
        cross = within.transpose(-1, -2)
        # Eq. (5): sum both heads and the sub-window input, then project.
        combined = self.head(within) + self.head(cross) + within
        out = self.output_projection(combined.flatten(-2)).transpose(1, 2)
        return restore_last_value(out, level)
