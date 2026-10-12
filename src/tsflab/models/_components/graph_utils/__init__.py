"""Canonical adjacency supports for spatiotemporal forecasting models.

This module is an independent composition of the repository's dense adjacency
normalizers and spectral helpers. It provides the small list-based interface
needed by graph forecasters without depending on an external implementation.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from tsflab.models._components.adj_norm import (
    reverse_transition_matrix,
    symmetric_normalized_laplacian,
    transition_matrix,
)
from tsflab.models._components.graph_spectral import chebyshev_polynomials, scaled_laplacian


def _adjacency(adj_mx: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adj_mx, dtype=np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adj_mx must be a square matrix")
    if not np.isfinite(adjacency).all():
        raise ValueError("adj_mx must contain only finite values")
    return adjacency


def _with_self_loops(adjacency: np.ndarray) -> list[np.ndarray]:
    with_self_loops = adjacency.copy()
    np.fill_diagonal(with_self_loops, 1.0)
    return [with_self_loops]


_SUPPORT_BUILDERS = {
    "normlap": lambda adjacency: [symmetric_normalized_laplacian(adjacency)],
    "scalap": lambda adjacency: [scaled_laplacian(adjacency)],
    "symadj": lambda adjacency: [
        np.eye(adjacency.shape[0])
        - symmetric_normalized_laplacian(adjacency)
    ],
    "transition": lambda adjacency: [transition_matrix(adjacency)],
    "doubletransition": lambda adjacency: [
        transition_matrix(adjacency), reverse_transition_matrix(adjacency)
    ],
    "identity": lambda adjacency: [np.eye(adjacency.shape[0])],
    "origin": _with_self_loops,
}


def normalize_adj_mx(
    adj_mx: np.ndarray, adj_type: str, return_type: str = "dense"
) -> list[np.ndarray | sp.coo_matrix]:
    """Return the requested finite adjacency supports.

    ``doubletransition`` yields forward and reverse random-walk matrices; all
    other modes yield one support. Dense outputs are float32 arrays and COO
    outputs contain the same values.
    """
    adjacency = _adjacency(adj_mx)
    builder = _SUPPORT_BUILDERS.get(adj_type)
    if builder is None:
        raise ValueError(f"unknown adjacency normalization: {adj_type}")
    supports = builder(adjacency)

    dense = [np.asarray(support, dtype=np.float32) for support in supports]
    if return_type == "dense":
        return dense
    if return_type == "coo":
        return [sp.coo_matrix(support) for support in dense]
    raise ValueError("return_type must be 'dense' or 'coo'")


def adj_to_supports(
    adj_mx: np.ndarray,
    adj_type: str = "doubletransition",
    device: str | torch.device = "cpu",
) -> list[torch.Tensor]:
    """Convert an adjacency matrix to dense float32 support tensors."""
    normalized = normalize_adj_mx(adj_mx, adj_type, return_type="dense")
    return [
        torch.as_tensor(support, dtype=torch.float32, device=device)
        for support in normalized
    ]


def cheb_poly(matrix: np.ndarray, order: int) -> np.ndarray:
    """Return exactly ``order`` Chebyshev polynomials, beginning with identity."""
    return chebyshev_polynomials(np.asarray(matrix), order)
