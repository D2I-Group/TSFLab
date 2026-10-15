"""Pluggable training callbacks and training tricks.

This module provides a small, opt-in callback system that lets training tricks
hook into the train loop without changing its default behavior. When no
callbacks are configured the train loop runs exactly as before (the hooks are
no-ops), so default runs remain byte-identical to the pre-callback trainer.

Hook contract
-------------
The trainer invokes the following hooks (all optional, all default no-op):

``on_train_start(ctx)``
    Called once before the first epoch.
``on_compute_loss(ctx) -> torch.Tensor | None``
    Called after the model forward and base-loss computation, before backward.
    May read/modify ``ctx.outputs`` / ``ctx.targets`` and return a replacement
    loss tensor. Returning ``None`` keeps ``ctx.loss`` unchanged. The trainer
    re-reads ``ctx.outputs``/``ctx.targets`` after this hook so callbacks that
    only reslice tensors (e.g. curriculum learning) do not need to recompute the
    loss themselves -- but if they do, returning it short-circuits the trainer's
    recompute.
``on_backward(ctx)``
    Called after ``loss.backward()`` and (under AMP) after the gradients have
    been unscaled, but before ``optimizer.step()``. The right place for gradient
    clipping.
``on_optimizer_step(ctx) -> bool``
    Called immediately before the trainer would call ``optimizer.step()``.
    Return ``False`` to skip the optimizer step (and the matching
    ``zero_grad``) for this micro-batch -- used by gradient accumulation.
    Returning ``True`` / ``None`` performs the step normally.
``on_epoch_end(ctx)``
    Called at the end of each epoch, after validation.

Gradient-accumulation contract
-------------------------------
``GradAccumulationCallback`` scales the loss by ``1/steps`` in
``on_compute_loss`` and only allows the optimizer to step every ``steps``
micro-batches via ``on_optimizer_step``. When the step is skipped the trainer
also skips ``zero_grad`` so gradients accumulate. The trainer flushes any
pending gradients at the end of each epoch (the callback signals readiness via
``ctx.is_last_batch``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn


@dataclass
class CallbackContext:
    """Mutable state passed to every callback hook.

    Attributes
    ----------
    model : nn.Module
        The model being trained.
    optimizer : torch.optim.Optimizer
        The optimizer instance.
    epoch : int
        Zero-based epoch index.
    total_epochs : int
        Total configured epochs.
    batch_idx : int
        Zero-based micro-batch index within the current epoch.
    is_last_batch : bool
        True when this is the final micro-batch of the epoch.
    pred_len : int
        Prediction horizon length.
    outputs : torch.Tensor | None
        Model outputs (already sliced to the forecast horizon).
    targets : torch.Tensor | None
        Ground-truth targets (already sliced to the forecast horizon).
    loss : torch.Tensor | None
        Current training loss tensor.
    extra : dict
        Free-form scratch space for callbacks.
    """

    model: nn.Module
    optimizer: torch.optim.Optimizer
    epoch: int = 0
    total_epochs: int = 0
    batch_idx: int = 0
    is_last_batch: bool = False
    pred_len: int = 0
    outputs: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    loss: torch.Tensor | None = None
    extra: dict = field(default_factory=dict)


class Callback:
    """Base callback with no-op hooks.

    Subclasses override only the hooks they need. All hooks default to no-ops so
    a callback list of arbitrary callbacks never changes default behavior.
    """

    def state_dict(self) -> dict:
        """Return scalar callback state for epoch-boundary recovery."""
        return dict(vars(self))

    def load_state_dict(self, state: dict) -> None:
        """Restore the callback's checkpoint state."""
        vars(self).update(state)

    def on_train_start(self, ctx: CallbackContext) -> None:
        """Called once before training begins."""

    def on_compute_loss(self, ctx: CallbackContext) -> torch.Tensor | None:
        """Optionally return a modified loss tensor (or None to keep ctx.loss)."""
        return None

    def on_backward(self, ctx: CallbackContext) -> None:
        """Called after backward, before the optimizer step."""

    def on_optimizer_step(self, ctx: CallbackContext) -> bool | None:
        """Return False to skip the optimizer step for this micro-batch."""
        return None

    def on_epoch_end(self, ctx: CallbackContext) -> None:
        """Called at the end of each epoch."""


class GradClipCallback(Callback):
    """Clip gradient norm before the optimizer step.

    Parameters
    ----------
    max_norm : float
        Maximum allowed gradient norm.
    norm_type : float
        The type of the used p-norm (default 2.0).
    """

    def __init__(self, max_norm: float, norm_type: float = 2.0) -> None:
        self.max_norm = max_norm
        self.norm_type = norm_type

    def on_backward(self, ctx: CallbackContext) -> None:
        torch.nn.utils.clip_grad_norm_(
            ctx.model.parameters(),
            max_norm=self.max_norm,
            norm_type=self.norm_type,
        )


class GradAccumulationCallback(Callback):
    """Accumulate gradients over ``steps`` micro-batches before stepping.

    The loss is scaled by ``1/steps`` so the accumulated gradient matches a
    single large batch. The optimizer only steps every ``steps`` micro-batches
    (or on the final batch of an epoch, to flush any remainder).

    Parameters
    ----------
    steps : int
        Number of micro-batches to accumulate per optimizer step.
    """

    def __init__(self, steps: int) -> None:
        if steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {steps}")
        self.steps = steps

    def on_compute_loss(self, ctx: CallbackContext) -> torch.Tensor | None:
        if self.steps == 1 or ctx.loss is None:
            return None
        return ctx.loss / self.steps

    def on_optimizer_step(self, ctx: CallbackContext) -> bool | None:
        if self.steps == 1:
            return None
        # Step every `steps` micro-batches, and always flush at epoch end.
        do_step = ((ctx.batch_idx + 1) % self.steps == 0) or ctx.is_last_batch
        return do_step


class CurriculumCallback(Callback):
    """Curriculum learning on the forecast horizon (BasicTS scheme).

    During the first ``warmup_epochs`` epochs the loss is computed on the full
    horizon. After warmup the effective horizon length grows linearly: every
    ``step_size`` epochs (counting from the end of warmup) the curriculum level
    advances by one, and the horizon used for the loss is
    ``min((level + 1) * cl_epochs, pred_len)``.

    Parameters
    ----------
    warmup_epochs : int
        Epochs of full-horizon training before the curriculum starts.
    step_size : int
        Number of epochs between curriculum level increments.
    cl_epochs : int
        Horizon increment (in time steps) per curriculum level.
    """

    def __init__(self, warmup_epochs: int, step_size: int, cl_epochs: int) -> None:
        if step_size < 1:
            raise ValueError(f"curriculum step_size must be >= 1, got {step_size}")
        if cl_epochs < 1:
            raise ValueError(f"curriculum cl_epochs must be >= 1, got {cl_epochs}")
        self.warmup_epochs = warmup_epochs
        self.step_size = step_size
        self.cl_epochs = cl_epochs

    def _cl_length(self, epoch: int, pred_len: int) -> int:
        if epoch < self.warmup_epochs:
            return pred_len
        level = (epoch - self.warmup_epochs) // self.step_size
        length = (level + 1) * self.cl_epochs
        return min(length, pred_len)

    def on_compute_loss(self, ctx: CallbackContext) -> torch.Tensor | None:
        if ctx.outputs is None or ctx.targets is None:
            return None
        cl_length = self._cl_length(ctx.epoch, ctx.pred_len)
        if cl_length >= ctx.pred_len:
            return None
        # Slice both tensors to the curriculum horizon. The trainer re-reads
        # ctx.outputs / ctx.targets after this hook and recomputes the loss.
        ctx.outputs = ctx.outputs[:, :cl_length, :]
        ctx.targets = ctx.targets[:, :cl_length, :]
        return None


_CALLBACK_FACTORIES = {
    "grad_clip": lambda v: GradClipCallback(
        max_norm=float(v if not isinstance(v, dict) else v.get("max_norm", 5.0)),
        norm_type=float(v.get("norm_type", 2.0)) if isinstance(v, dict) else 2.0,
    ),
    "grad_accum": lambda v: GradAccumulationCallback(steps=int(v if not isinstance(v, dict) else v.get("steps", 1))),
    "curriculum": lambda v: CurriculumCallback(
        warmup_epochs=int(v.get("warmup_epochs", 0)),
        step_size=int(v.get("step_size", 1)),
        cl_epochs=int(v.get("cl_epochs", 1)),
    ),
}


def build_callbacks(tricks: Any) -> list[Callback]:
    """Build a callback list from a ``TrainTricksConfig`` (or None).

    Returns an empty list when no tricks are configured, which makes the trainer
    behave exactly as before. The construction order is fixed and meaningful:
    gradient accumulation scales the loss first, curriculum reslices, gradient
    clipping runs in ``on_backward``.

    Parameters
    ----------
    tricks : TrainTricksConfig | None
        Validated tricks config, or None.

    Returns
    -------
    list[Callback]
        Concrete callbacks to attach to the train loop.
    """
    if tricks is None:
        return []

    callbacks: list[Callback] = []

    grad_accum_steps = getattr(tricks, "grad_accum_steps", 1)
    if grad_accum_steps and grad_accum_steps > 1:
        callbacks.append(GradAccumulationCallback(steps=int(grad_accum_steps)))

    curriculum = getattr(tricks, "curriculum", None)
    if curriculum is not None and getattr(curriculum, "enabled", False):
        callbacks.append(
            CurriculumCallback(
                warmup_epochs=curriculum.warmup_epochs,
                step_size=curriculum.step_size,
                cl_epochs=curriculum.cl_epochs,
            )
        )

    grad_clip_norm = getattr(tricks, "grad_clip_norm", None)
    if grad_clip_norm is not None and grad_clip_norm > 0:
        callbacks.append(
            GradClipCallback(
                max_norm=float(grad_clip_norm),
                norm_type=float(getattr(tricks, "grad_clip_norm_type", 2.0)),
            )
        )

    return callbacks
