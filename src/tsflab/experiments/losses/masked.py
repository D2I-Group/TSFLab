"""Masked regression losses (mae / mse / rmse).

These criteria operate on ``(prediction, target)`` tensors of shape
``(B, pred_len, N)`` — the pair TSFLab's trainer passes to the
criterion — and accept an OPTIONAL ``targets_mask`` (1 = valid, 0 = ignore)
of the same shape.

When the mask is ``None`` (the default, and what the current 2-arg trainer
call produces) each loss is EXACTLY equivalent to the plain ``mae`` / ``mse``
/ ``rmse`` over all elements, so default training is unchanged.

Following the BasicTS convention, a provided mask is normalized by its own
mean (``mask = mask / mask.mean()``; ``nan_to_num``) so the loss stays
unbiased with respect to the number of valid entries.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.catalog.registry import LOSS_REGISTRY


def _normalize_mask(mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """Normalize a validity mask by its own mean (BasicTS convention).

    The mask is cast to ``like``'s dtype/device, divided by its mean so the
    expected per-element weight is 1.0, and any NaN/inf introduced by an
    all-zero mask is replaced with 0.
    """
    mask = mask.to(dtype=like.dtype, device=like.device)
    mask = mask / mask.mean()
    return torch.nan_to_num(mask)


class MaskedMAELoss(nn.Module):
    """Mean absolute error with an optional validity mask.

    With ``targets_mask=None`` this equals ``nn.L1Loss()`` over all elements.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        targets_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the (optionally masked) mean absolute error."""
        loss = torch.abs(pred - target)
        if targets_mask is None:
            return loss.mean()
        loss = loss * _normalize_mask(targets_mask, loss)
        return loss.mean()


class MaskedMSELoss(nn.Module):
    """Mean squared error with an optional validity mask.

    With ``targets_mask=None`` this equals ``nn.MSELoss()`` over all elements.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        targets_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the (optionally masked) mean squared error."""
        loss = (pred - target) ** 2
        if targets_mask is None:
            return loss.mean()
        loss = loss * _normalize_mask(targets_mask, loss)
        return loss.mean()


class MaskedRMSELoss(nn.Module):
    """Root mean squared error with an optional validity mask.

    With ``targets_mask=None`` this equals ``sqrt(MSE)`` over all elements.
    """

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        targets_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the (optionally masked) root mean squared error."""
        loss = (pred - target) ** 2
        if targets_mask is None:
            return torch.sqrt(loss.mean())
        loss = loss * _normalize_mask(targets_mask, loss)
        return torch.sqrt(loss.mean())


def register() -> None:
    """Register masked regression losses into the registry."""
    LOSS_REGISTRY.register("masked_mae", lambda **kwargs: MaskedMAELoss())
    LOSS_REGISTRY.register("masked_mse", lambda **kwargs: MaskedMSELoss())
    LOSS_REGISTRY.register("masked_rmse", lambda **kwargs: MaskedRMSELoss())
