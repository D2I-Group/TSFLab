"""Resumable optimization stage for model-owned reconstruction pretraining."""

import os
from pathlib import Path
import tempfile

import torch

from tsflab.experiments.infra.checkpoint import capture_rng, set_rng


def atomic_state(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".stage-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _load_stage(checkpoint, module, optimizer, loader):
    """Restore a saved stage into the module, optimizer, RNG, and loader; return its state or None."""
    state = None
    if checkpoint is not None and Path(checkpoint).exists():
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        module.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        set_rng(state["rng"])
        if state.get("current_loader_rng") is not None:
            loader.generator.set_state(state["current_loader_rng"])
    return state


def _resume_epoch(state, epoch, loader):
    """Batches to skip in ``epoch`` and its RNG snapshots: (skip, current_rng, epoch_rng, loader_rng)."""
    progress = state if state and state["epoch"] == epoch else None
    skip = progress["next_batch"] if progress else 0
    current_rng = capture_rng()
    if progress:
        set_rng(progress["epoch_rng"])
        if progress["loader_rng"] is not None:
            loader.generator.set_state(progress["loader_rng"])
    epoch_rng = capture_rng()
    loader_rng = (
        loader.generator.get_state()
        if getattr(loader, "generator", None) is not None
        else None
    )
    return skip, current_rng, epoch_rng, loader_rng


def _stage_batches(
    module, loader, device, optimizer, criterion, epoch, *, checkpoint, every_batches, skip, current_rng,
    epoch_rng, loader_rng,
):
    """Optimize one epoch, skipping replayed batches and checkpointing every ``every_batches`` steps."""
    for index, batch in enumerate(loader):
        if index < skip:
            if index + 1 == skip:
                set_rng(current_rng)
            continue
        values = batch[0].float().to(device)
        optimizer.zero_grad()
        criterion(module(values), values).backward()
        optimizer.step()
        if (
            checkpoint is not None
            and every_batches
            and (index + 1) % every_batches == 0
        ):
            atomic_state(
                checkpoint,
                {
                    "model": module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "next_batch": index + 1,
                    "rng": capture_rng(),
                    "epoch_rng": epoch_rng,
                    "loader_rng": loader_rng,
                },
            )


def _boundary_state(module, optimizer, loader, epoch) -> dict:
    """Stage state at the start of ``epoch``, before any of its batches."""
    return {
        "model": module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "next_batch": 0,
        "rng": capture_rng(),
        "epoch_rng": capture_rng(),
        "loader_rng": None,
        "current_loader_rng": loader.generator.get_state()
        if getattr(loader, "generator", None) is not None
        else None,
    }


def reconstruction_stage(
    module,
    loader,
    device,
    optimizer,
    criterion,
    epochs,
    *,
    checkpoint=None,
    every_batches=0,
):
    """Resume after a complete optimizer step; ordinary model use needs no path."""
    state = _load_stage(checkpoint, module, optimizer, loader)
    if checkpoint is not None and every_batches and getattr(loader, "num_workers", 0):
        raise ValueError("resumable pretraining requires num_workers=0")
    start = state["epoch"] if state else 0
    for epoch in range(start, epochs):
        skip, current_rng, epoch_rng, loader_rng = _resume_epoch(state, epoch, loader)
        _stage_batches(
            module, loader, device, optimizer, criterion, epoch,
            checkpoint=checkpoint, every_batches=every_batches, skip=skip, current_rng=current_rng,
            epoch_rng=epoch_rng, loader_rng=loader_rng,
        )
        if checkpoint is not None:
            atomic_state(checkpoint, _boundary_state(module, optimizer, loader, epoch + 1))
        state = None
    if checkpoint is not None:
        atomic_state(checkpoint, _boundary_state(module, optimizer, loader, epochs))
