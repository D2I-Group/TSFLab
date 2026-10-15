"""Canonical tensor boundary between runners and forecasting models."""

from __future__ import annotations

import torch
import torch.nn as nn


def unwrap_model(model: nn.Module) -> nn.Module:
    """Return the underlying module when DataParallel wraps a forecaster."""
    return model.module if isinstance(model, nn.DataParallel) else model


def make_decoder_input(
    batch_y: torch.Tensor,
    label_len: int,
    pred_len: int,
    device: torch.device,
) -> torch.Tensor:
    """Join the observed decoder label window with a zero future horizon."""
    zeros = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
    return torch.cat([batch_y[:, :label_len, :], zeros], dim=1).float().to(device)


def call_forecaster(
    model: nn.Module,
    batch_x: torch.Tensor,
    batch_x_mark: torch.Tensor | None,
    decoder_input: torch.Tensor,
    batch_y_mark: torch.Tensor | None,
) -> torch.Tensor:
    """Call the one public forecasting interface.

    Every catalog model accepts encoder values, encoder marks, decoder values,
    and decoder marks in that order. Models that do not use an optional input
    ignore it in their own local implementation; the runner never guesses a
    calling convention from a Python signature.
    """
    return model(batch_x, batch_x_mark, decoder_input, batch_y_mark)


def slice_prediction_target(
    outputs: torch.Tensor,
    batch_y: torch.Tensor,
    pred_len: int,
    features: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align point or probabilistic output with the forecast target tensor.

    Point outputs have shape ``(B, L, C)``. Quantile and distribution outputs
    add a final parameter axis ``(B, L, C, K)`` that must remain intact while
    the horizon and optional final target channel are selected.
    """
    feature_start = -1 if features == "MS" else 0
    if outputs.dim() == 4:
        outputs = outputs[:, -pred_len:, feature_start:, :]
    else:
        outputs = outputs[:, -pred_len:, feature_start:]
    target = batch_y[:, -pred_len:, feature_start:]
    return outputs, target


def slice_target(batch_y: torch.Tensor, pred_len: int, features: str) -> torch.Tensor:
    """Select the canonical forecast target without requiring model output."""
    feature_start = -1 if features == "MS" else 0
    return batch_y[:, -pred_len:, feature_start:]
