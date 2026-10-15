"""Typed contract for model-provided training objectives.

A model that needs a paper-specific loss declares it on ``ModelSpec`` as
``training_objective`` (and optionally ``training_setup``). The trainer calls it
only while training; validation, early stopping and test metrics always use the
configured observation criterion on ``forward`` outputs. Models that declare
nothing train exactly as before (configured criterion plus the optional
``aux_loss`` regularizer).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch
import torch.nn as nn

from tsflab.experiments.runner.model_io import (
    call_forecaster,
    slice_prediction_target,
    slice_target,
)


@dataclass(frozen=True)
class TrainingBatch:
    """One training micro-batch as seen by a training objective."""

    x: torch.Tensor
    x_mark: torch.Tensor | None
    dec_inp: torch.Tensor
    y_mark: torch.Tensor | None
    y: torch.Tensor
    pred_len: int
    features: str

    @property
    def target(self) -> torch.Tensor:
        """Forecast target ``(B, pred_len, C)`` (last channel only for ``MS``)."""
        return slice_target(self.y, self.pred_len, self.features)

    def forecast(self, model: nn.Module) -> torch.Tensor:
        """Run the standard four-input forward pass."""
        return call_forecaster(model, self.x, self.x_mark, self.dec_inp, self.y_mark)

    def align(self, outputs: torch.Tensor) -> torch.Tensor:
        """Select the horizon and target channels of ``outputs``."""
        return slice_prediction_target(outputs, self.y, self.pred_len, self.features)[0]


class TrainingObjective(Protocol):
    """``(model, batch, criterion) -> (forecast | None, scalar loss)``.

    ``model`` is the unwrapped module in training mode and ``criterion`` the
    configured observation loss, available to objectives that extend it
    (``loss:<name>+local:<term>``). ``forecast`` is the point forecast produced
    by the same pass; return ``None`` when the objective does not produce one
    (the trainer then re-runs ``forward`` without gradients only if a callback
    needs outputs). The loss must be a finite scalar tensor.
    """

    def __call__(
        self, model: nn.Module, batch: TrainingBatch, criterion: Callable
    ) -> tuple[torch.Tensor | None, torch.Tensor]: ...


class TrainingSetup(Protocol):
    """``(model, train_loader, *, pred_len, features) -> None``.

    Optional one-time preparation from the training split (for example fitting a
    label basis) run before the first epoch of a fresh training run.
    """

    def __call__(self, model: nn.Module, train_loader, *, pred_len: int, features: str) -> None: ...
