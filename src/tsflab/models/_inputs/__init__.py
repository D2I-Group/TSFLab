"""Shared runtime checks of the forecasting input contract.

Constructor parameters are validated once by each model's params schema (``spec.py``) and
task values by ``TaskConfig``. What remains for model code is the shape of the tensors it
receives; these helpers state that contract once, so every model raises the same error.
"""

from __future__ import annotations

import numpy as np
import torch


def check_window(x: torch.Tensor, length: int, channels: int, name: str = "x_enc") -> torch.Tensor:
    """Return ``x`` if it is ``[batch, length, channels]``; otherwise raise ``ValueError``."""
    if x.ndim != 3 or tuple(x.shape[1:]) != (length, channels):
        raise ValueError(f"{name} must have shape (batch, {length}, {channels}), got {tuple(x.shape)}")
    return x


def check_marks(marks: torch.Tensor | None, x: torch.Tensor, columns: int = 6,
                name: str = "x_mark_enc") -> torch.Tensor | None:
    """Return ``marks`` if absent or ``[batch, time, columns]`` aligned with ``x``; otherwise raise."""
    if marks is not None and (marks.ndim != 3 or marks.shape[:2] != x.shape[:2] or marks.shape[-1] != columns):
        raise ValueError(f"{name} must have shape ({x.shape[0]}, {x.shape[1]}, {columns}), got {tuple(marks.shape)}")
    return marks


def square_adjacency(adj, nodes: int, dtype=np.float32, name: str = "adj_mx") -> np.ndarray:
    """``adj`` as a ``[nodes, nodes]`` numpy array of ``dtype``; raise ``ValueError`` on another shape."""
    adjacency = np.asarray(adj, dtype=dtype)
    if adjacency.shape != (nodes, nodes):
        raise ValueError(f"{name} must have shape {(nodes, nodes)}, got {adjacency.shape}")
    return adjacency
