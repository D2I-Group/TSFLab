"""Robust scaled-Laplacian and Chebyshev supports for spectral graph models."""

from __future__ import annotations

import numpy as np
import torch


def scaled_laplacian(adj_mx: np.ndarray, *, undirected: bool = True) -> np.ndarray:
    """Return a dense scaled normalized Laplacian for any finite square graph."""
    adjacency = np.asarray(adj_mx, dtype=np.float64)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adj_mx must be a square matrix")
    if not np.isfinite(adjacency).all():
        raise ValueError("adj_mx must contain only finite values")
    if undirected:
        adjacency = np.maximum(adjacency, adjacency.T)
    degree = adjacency.sum(axis=1)
    inverse = np.zeros_like(degree)
    positive = degree > 0
    inverse[positive] = degree[positive] ** -0.5
    laplacian = np.eye(adjacency.shape[0]) - inverse[:, None] * adjacency * inverse[None]
    if laplacian.size == 0:
        return laplacian.astype(np.float32)
    eigenvalues = np.linalg.eigvalsh(laplacian) if undirected else np.linalg.eigvals(laplacian)
    lambda_max = float(np.max(np.abs(eigenvalues)))
    identity = np.eye(adjacency.shape[0], dtype=laplacian.dtype)
    scaled = laplacian - identity if lambda_max < 1e-12 else 2.0 * laplacian / lambda_max - identity
    return np.asarray(np.real_if_close(scaled), dtype=np.float32)


def chebyshev_polynomials(matrix: np.ndarray, order: int) -> np.ndarray:
    """Return exactly ``order`` Chebyshev polynomials, beginning with identity."""
    matrix = np.asarray(matrix)
    if order < 1:
        raise ValueError("order must be at least 1")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    polynomials = [np.eye(matrix.shape[0], dtype=matrix.dtype)]
    if order > 1:
        polynomials.append(matrix.copy())
    for index in range(2, order):
        polynomials.append(2.0 * matrix @ polynomials[index - 1] - polynomials[index - 2])
    return np.asarray(polynomials)


def chebyshev_supports(adj_mx: np.ndarray, order: int, *, undirected: bool = True) -> torch.Tensor:
    """Build exactly ``order`` dense Chebyshev supports for an adjacency matrix."""
    basis = chebyshev_polynomials(scaled_laplacian(adj_mx, undirected=undirected), order)
    return torch.as_tensor(basis, dtype=torch.float32)
