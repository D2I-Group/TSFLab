"""Specialized loss functions used by local forecasting models.

These implement frequency-domain objectives described by the corresponding
papers and reference training scripts, expressed as standard
``nn.Module`` criteria that operate on ``(prediction, target)`` tensors of
shape ``(B, pred_len, N)`` — the exact pair TSFLab's trainer passes to
the criterion.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.catalog.registry import LOSS_REGISTRY


class FrequencyMAELoss(nn.Module):
    """Mean absolute error in the temporal frequency domain.

    Computes the real FFT of prediction and target along the time axis and
    averages the absolute difference of the complex spectra. This matches
    AirCade's training objective
    ``(rfft(pred) - rfft(target)).abs().mean()``.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the frequency-domain MAE along the time dimension."""
        pred_f = torch.fft.rfft(pred, dim=1)
        target_f = torch.fft.rfft(target, dim=1)
        return (pred_f - target_f).abs().mean()


class FrequencyWeightedMAELoss(nn.Module):
    """Self-normalized frequency-domain MAE used by MoFo.

    For each channel a per-channel frequency MAE ``loss2`` is computed, then
    rescaled by ``loss2 / loss2.detach() * loss2.detach().max()`` and summed.
    The rescaling keeps gradients balanced across channels while preserving
    the overall magnitude of the dominant channel.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return the self-normalized frequency-weighted MAE."""
        pred_f = torch.fft.rfft(pred, dim=1)
        target_f = torch.fft.rfft(target, dim=1)
        # Per-channel mean over batch and frequency bins -> (N,)
        loss2 = (pred_f - target_f).abs().mean(dim=(0, 1))
        denom = loss2.detach()
        scale = denom.max()
        return ((loss2 / (denom + 1e-12)) * scale).sum()


def register() -> None:
    """Register specialized forecasting losses into the registry."""
    LOSS_REGISTRY.register("freq_mae", lambda **kwargs: FrequencyMAELoss())
    LOSS_REGISTRY.register("freq_weighted_mae", lambda **kwargs: FrequencyWeightedMAELoss())
