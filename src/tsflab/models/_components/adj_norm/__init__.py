"""Adjacency-matrix normalization utilities for graph forecasting models.

Pure-numpy (no torch dependency) implementations of the standard adjacency
normalizations used by spatiotemporal GNNs. The definitions are also used by
the Apache-2.0 BasicTS ``basicts/utils/adjacent_matrix_norm.py``:
https://github.com/GestaltCogTeam/BasicTS (Apache-2.0). The math matches the
published reference definitions; this local implementation operates on dense numpy
``(N, N)`` arrays and guards against zero-degree rows.

All functions accept a dense adjacency ``adj`` of shape ``(N, N)`` and return a
dense ``(N, N)`` ``float64`` matrix. Zero-degree nodes are handled by treating
their inverse degree as ``0`` (rather than ``inf``), so the result is finite.
"""

from __future__ import annotations

import numpy as np


def _as_dense(adj) -> np.ndarray:
    """Coerce ``adj`` to a dense 2-D float64 numpy array and validate shape."""
    a = np.asarray(adj, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"adjacency must be a square (N, N) matrix, got {a.shape}")
    return a


def _inv_sqrt_degree(deg: np.ndarray) -> np.ndarray:
    """D^{-1/2} as a dense diagonal matrix, zero for zero-degree rows."""
    with np.errstate(divide="ignore"):
        d = np.power(deg, -0.5)
    d[np.isinf(d)] = 0.0
    return np.diag(d)


def _inv_degree(deg: np.ndarray) -> np.ndarray:
    """D^{-1} as a dense diagonal matrix, zero for zero-degree rows."""
    with np.errstate(divide="ignore"):
        d = np.power(deg, -1.0)
    d[np.isinf(d)] = 0.0
    return np.diag(d)


def symmetric_normalized_laplacian(adj) -> np.ndarray:
    """Symmetric normalized Laplacian ``L = I - D^{-1/2} A D^{-1/2}``.

    Degrees are computed from the row-sums of ``A``.
    """
    a = _as_dense(adj)
    n = a.shape[0]
    deg = a.sum(axis=1)
    d_inv_sqrt = _inv_sqrt_degree(deg)
    return np.eye(n) - d_inv_sqrt @ a @ d_inv_sqrt


def lambda_rescaled_laplacian(adj, lambda_max: float = 2.0) -> np.ndarray:
    """Scaled Laplacian ``2L / lambda_max - I`` for Chebyshev polynomials.

    ``L`` is the symmetric normalized Laplacian. With the default
    ``lambda_max = 2`` this reduces to ``L - I``. Pass the true largest
    eigenvalue of ``L`` for an exact rescaling onto ``[-1, 1]``.
    """
    a = _as_dense(adj)
    n = a.shape[0]
    lap = symmetric_normalized_laplacian(a)
    return (2.0 / lambda_max) * lap - np.eye(n)


def gcn_norm(adj) -> np.ndarray:
    """GCN renormalization ``D^{-1/2} (A + I) D^{-1/2}`` (Kipf & Welling).

    Self-loops are added before computing degrees.
    """
    a = _as_dense(adj)
    n = a.shape[0]
    a_loop = a + np.eye(n)
    deg = a_loop.sum(axis=1)
    d_inv_sqrt = _inv_sqrt_degree(deg)
    return d_inv_sqrt @ a_loop @ d_inv_sqrt


def transition_matrix(adj) -> np.ndarray:
    """Random-walk transition matrix ``D^{-1} A`` (row-normalized)."""
    a = _as_dense(adj)
    deg = a.sum(axis=1)
    d_inv = _inv_degree(deg)
    return d_inv @ a


def reverse_transition_matrix(adj) -> np.ndarray:
    """Reverse random-walk transition matrix ``(D^{-1} A)`` on ``A^T``.

    Equivalent to ``transition_matrix`` applied to the transposed adjacency,
    i.e. ``D_in^{-1} A^T`` where ``D_in`` is the in-degree diagonal.
    """
    a = _as_dense(adj)
    return transition_matrix(a.T)
