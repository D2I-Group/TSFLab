"""Probabilistic loss functions for Phase 1 forecasting (quantile / Gaussian NLL).

These criteria operate on the rank-4 probabilistic prediction tensors produced
by probabilistic models (``output_type != "point"``) against the rank-3 target
``(B, pred_len, C)`` the trainer passes through ``_slice_pred_target``:

- ``QuantileLoss`` consumes a quantile grid ``(B, L, C, Q)`` and the rank-3
  target, returning the weighted pinball loss averaged over all elements and
  quantile levels.
- ``GaussianNLLLoss`` consumes a ``(B, L, C, 2) = (loc, scale)`` tensor and the
  rank-3 target, returning the mean Gaussian negative log-likelihood.

The point losses (``mse`` / ``mae``) in :mod:`tsflab.experiments.losses` are
unchanged and remain the default.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.catalog.registry import LOSS_REGISTRY


class QuantileLoss(nn.Module):
    """Weighted pinball (quantile) loss over a grid of quantile levels.

    For each quantile level ``q`` and element the pinball loss is
    ``max(q * (y - yhat_q), (q - 1) * (y - yhat_q))``; the final loss is the
    mean over batch, horizon, channels, and quantile levels.
    """

    def __init__(self, quantile_levels: list[float] | None = None) -> None:
        super().__init__()
        levels = quantile_levels or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.q = torch.tensor(levels, dtype=torch.float32)

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Return the weighted pinball loss.

        Parameters
        ----------
        pred : torch.Tensor
            Quantile predictions of shape ``(B, L, C, Q)``, last axis ordered
            identically to the configured quantile levels.
        true : torch.Tensor
            Ground-truth target of shape ``(B, L, C)``.
        """
        y = true.unsqueeze(-1)  # (B, L, C, 1)
        q = self.q.to(pred.device).view(1, 1, 1, -1)  # (1, 1, 1, Q)
        assert pred.shape[-1] == q.shape[-1]
        e = y - pred
        loss = torch.maximum(q * e, (q - 1.0) * e)
        return loss.mean()


class GaussianNLLLoss(nn.Module):
    """Gaussian negative log-likelihood loss for ``(loc, scale)`` predictions.

    Computes ``0.5 * log(2 * pi * scale^2) + 0.5 * ((y - loc) / scale)^2``
    elementwise, mean-reduced over batch, horizon, and channels.
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Return the mean Gaussian negative log-likelihood.

        Parameters
        ----------
        pred : torch.Tensor
            Distribution parameters of shape ``(B, L, C, 2)`` where
            ``[..., 0]`` is the location and ``[..., 1]`` the (positive) scale.
        true : torch.Tensor
            Ground-truth target of shape ``(B, L, C)``.
        """
        loc = pred[..., 0]
        scale = pred[..., 1].clamp_min(self.eps)
        nll = 0.5 * torch.log(2 * math.pi * scale**2) + 0.5 * ((true - loc) / scale) ** 2
        return nll.mean()


def register() -> None:
    """Register probabilistic losses into the registry."""
    LOSS_REGISTRY.register("quantile", lambda **kwargs: QuantileLoss(**kwargs))
    LOSS_REGISTRY.register("nll_gaussian", lambda **kwargs: GaussianNLLLoss(**kwargs))
