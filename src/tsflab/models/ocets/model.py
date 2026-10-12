"""OCE-TS: ordinal classification over value bins trained with ordinal cross-entropy.

Independent implementation from the paper (Methodology section, Eqs. 2 and 8-18,
Appendix "Data Normalization", Eqs. 74-77) after reading the pinned official code
(``Shi-hm/OCE-TS`` at ``a574b99e``) to resolve the head layout; nothing is copied.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

from tsflab.models._components.dlinear import DLinearBackbone
from tsflab.models._inputs import check_window

ValueRange = Literal["signed", "unit"]
_NORM_EPS = 1e-8


def minmax_normalize(values: torch.Tensor, value_range: ValueRange) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Eqs. (74)/(76): per-sample, per-channel min-max scaling over the time axis.

    Returns the scaled ``[B, T, C]`` tensor and the ``[B, 1, C]`` minimum and maximum.
    """
    low = values.amin(dim=1, keepdim=True)
    high = values.amax(dim=1, keepdim=True)
    unit = (values - low) / (high - low + _NORM_EPS)
    return (2.0 * unit - 1.0 if value_range == "signed" else unit), low, high


def minmax_denormalize(values: torch.Tensor, low: torch.Tensor, high: torch.Tensor, value_range: ValueRange) -> torch.Tensor:
    """Inverse of :func:`minmax_normalize` (Eqs. 75/77) with the given statistics."""
    unit = (values + 1.0) * 0.5 if value_range == "signed" else values
    return unit * (high - low + _NORM_EPS) + low


def truncated_gaussian_bin_probabilities(target: torch.Tensor, edges: torch.Tensor, sigma: float) -> torch.Tensor:
    """Target-to-Probability Transformation (Eqs. 8-11).

    A Gaussian centered at each target value, truncated to ``[edges[0], edges[-1]]``,
    integrated over every bin ``[l_k, u_k)``. ``target`` is any shape; the result
    adds a trailing bin axis of size ``len(edges) - 1`` that sums to one.
    """
    scaled = (edges - target.unsqueeze(-1)) / (sigma * math.sqrt(2.0))
    cdf = torch.erf(scaled)
    mass = cdf[..., -1] - cdf[..., 0]  # 2Z in Eq. (9)
    return (cdf[..., 1:] - cdf[..., :-1]) / mass.clamp_min(1e-10).unsqueeze(-1)


def ordinal_cross_entropy(pred_probs: torch.Tensor, true_probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Eq. (2): binary cross-entropy between cumulative distributions at K - 1 cut points.

    Summed over the cut points and averaged over every other axis. Cumulative
    probabilities are clamped to ``[eps, 1 - eps]`` and ``eps`` is also added inside
    each logarithm (the official numerical guard; it keeps float32 finite when a
    clamped cumulative probability rounds to one).
    """
    if pred_probs.shape != true_probs.shape:
        raise ValueError(f"shape mismatch: {tuple(pred_probs.shape)} vs {tuple(true_probs.shape)}")
    pred_cdf = pred_probs.cumsum(-1)[..., :-1].clamp(eps, 1.0 - eps)
    true_cdf = true_probs.cumsum(-1)[..., :-1].clamp(eps, 1.0 - eps)
    per_cut = true_cdf * (pred_cdf + eps).log() + (1.0 - true_cdf) * (1.0 - pred_cdf + eps).log()
    return -per_cut.sum(-1).mean()


class Model(nn.Module):
    """DLinear ordinal classifier; ``forward`` returns the probability-weighted bin centers."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        num_bins: int = 45,
        sigma: float = 0.015,
        kernel_size: int = 25,
        individual: bool = False,
        value_range: ValueRange = "signed",
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_bins = num_bins
        self.sigma = sigma
        self.value_range = value_range
        # Deep ordinal classifier: DLinear projections straight to pred_len * K logits
        # per channel (official layout), softmax over the K bins (Eq. 14).
        self.backbone = DLinearBackbone(enc_in, seq_len, pred_len * num_bins, kernel_size, individual)
        low = -1.0 if value_range == "signed" else 0.0
        edges = torch.linspace(low, 1.0, num_bins + 1)
        self.register_buffer("edges", edges, persistent=False)
        self.register_buffer("centers", (edges[1:] + edges[:-1]) / 2, persistent=False)

    def _validate(self, x_enc: torch.Tensor) -> None:
        check_window(x_enc, self.seq_len, self.enc_in)

    def bin_probabilities(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicted ``[B, pred_len, C, K]`` bin probabilities and the input min/max."""
        self._validate(x_enc)
        normalized, low, high = minmax_normalize(x_enc, self.value_range)
        logits = self.backbone(normalized)  # [B, pred_len * K, C]
        batch = logits.shape[0]
        logits = logits.transpose(1, 2).reshape(batch, self.enc_in, self.pred_len, self.num_bins)
        return logits.permute(0, 2, 1, 3).softmax(-1), low, high

    def predict_distribution(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Categorical forecast: probabilities ``[B, H, C, K]`` and bin centers ``[B, 1, C, K]``
        mapped back to the input scale with the lookback min/max."""
        probs, low, high = self.bin_probabilities(x_enc)
        centers = minmax_denormalize(self.centers, low.unsqueeze(-1), high.unsqueeze(-1), self.value_range)
        return probs, centers

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        del x_mark_enc, x_dec, x_mark_dec
        probs, low, high = self.bin_probabilities(x_enc)
        expected = (probs * self.centers).sum(-1)  # Eq. (18)
        return minmax_denormalize(expected, low, high, self.value_range)

    def soft_labels(self, target: torch.Tensor) -> torch.Tensor:
        """Truncated-Gaussian bin labels of a ``[B, H, C]`` target scaled by its own min/max."""
        normalized, _, _ = minmax_normalize(target, self.value_range)
        return truncated_gaussian_bin_probabilities(normalized, self.edges, self.sigma)

    def training_objective(self, x_enc: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Eq. (15): OCE between predicted bins and soft labels; also returns the forecast."""
        probs, low, high = self.bin_probabilities(x_enc)
        forecast = minmax_denormalize((probs * self.centers).sum(-1), low, high, self.value_range)
        horizon = probs[:, -target.shape[1]:, -target.shape[2]:]
        return forecast, ordinal_cross_entropy(horizon, self.soft_labels(target))
