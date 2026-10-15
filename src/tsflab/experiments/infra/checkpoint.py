"""Complete, atomic epoch-boundary checkpoints for local training recovery."""

import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch


def save_checkpoint(
    path,
    *,
    model,
    optimizer,
    scaler,
    early_stopping,
    manager,
    callbacks,
    epoch,
    elapsed,
    completed=False,
    loaders=(),
    progress=None,
):
    """Save only at an optimizer/epoch boundary, with no pending gradients."""
    rng = capture_rng()
    payload = {
        "schema_version": 1,
        "progress": progress,
        "model": model.state_dict(),
        "runtime_state": runtime_state(model),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler else None,
        "early_stopping": vars(early_stopping),
        "top_checkpoints": manager.top_checkpoints,
        "epoch": epoch,
        "elapsed": elapsed,
        "completed": completed,
        "rng": rng,
        "callbacks": [cb.state_dict() for cb in callbacks],
        "loader_rng": [
            loader.generator.get_state()
            if getattr(loader, "generator", None) is not None
            else None
            for loader in loaders
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".checkpoint-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _check_resume_contract(state, callbacks) -> None:
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != 1
        or "optimizer" not in state
    ):
        raise ValueError("checkpoint is not a complete TSFLab resume checkpoint")
    if len(state["callbacks"]) != len(callbacks):
        raise ValueError("checkpoint callback contract changed")


def _restore_loader_rng(loaders, saved_states) -> None:
    for loader, saved in zip(loaders, saved_states):
        if saved is not None:
            if getattr(loader, "generator", None) is None:
                raise ValueError("checkpoint requires a DataLoader generator")
            loader.generator.set_state(saved)


def restore_checkpoint(
    path, *, model, optimizer, scaler, early_stopping, manager, callbacks, loaders=()
):
    """Restore trusted local state, including RNG after all constructors have run."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    _check_resume_contract(state, callbacks)
    model.load_state_dict(state["model"], strict=True)
    restore_runtime_state(model, state.get("runtime_state"))
    optimizer.load_state_dict(state["optimizer"])
    if scaler and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    vars(early_stopping).update(state["early_stopping"])
    manager.top_checkpoints = state["top_checkpoints"]
    for cb, saved in zip(callbacks, state["callbacks"]):
        cb.load_state_dict(saved)
    _restore_loader_rng(loaders, state.get("loader_rng", []))
    set_rng(state["rng"])
    return state


def save_weights(path, model):
    """Keep the previous best weights intact until the new file is complete."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=".weights-")
    try:
        with os.fdopen(descriptor, "wb") as stream:
            torch.save(model.state_dict(), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def capture_rng():
    rng = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        rng["mps"] = torch.mps.get_rng_state()
    return rng


def set_rng(rng):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if "cuda" in rng and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])
    if "mps" in rng and torch.backends.mps.is_available():
        torch.mps.set_rng_state(rng["mps"])


def runtime_state(model):
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    save = getattr(target, "runtime_state_dict", None)
    restore = getattr(target, "load_runtime_state_dict", None)
    if callable(save) != callable(restore):
        raise ValueError("external runtime must implement both state hooks")
    return {"state": save()} if callable(save) else None


def restore_runtime_state(model, state):
    target = model.module if isinstance(model, torch.nn.DataParallel) else model
    restore = getattr(target, "load_runtime_state_dict", None)
    if state is not None:
        if not callable(restore):
            raise ValueError("checkpoint requires external runtime restoration hook")
        restore(state["state"])
