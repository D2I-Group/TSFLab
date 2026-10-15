---
name: "adj_norm"
description: "NumPy normalizations of a dense static adjacency: normalized and rescaled Laplacian, GCN renormalization, forward/reverse random-walk transitions. Use for preparing a small dataset graph; not for large sparse graphs, learned or batched adjacencies (not differentiable), or ready supports (use graph_utils)."
---

# adj_norm

## What it does

Pure-numpy helpers that turn a dense adjacency matrix `A` into the matrices that
spatiotemporal GNN layers consume. With `D` the diagonal of row sums:

- `symmetric_normalized_laplacian`: `L = I - D^{-1/2} A D^{-1/2}`.
- `lambda_rescaled_laplacian`: `(2 / lambda_max) * L - I` (Chebyshev rescaling).
- `gcn_norm`: `D~^{-1/2} (A + I) D~^{-1/2}` with degrees of `A + I`.
- `transition_matrix`: `D^{-1} A` (row-stochastic random walk).
- `reverse_transition_matrix`: `transition_matrix(A.T)`.

Zero-degree rows get inverse degree 0, so results stay finite (an isolated
node has Laplacian diagonal 1, transition row 0).

## When to use

Use for small dense static graphs when a numpy-side normalization is needed
before converting to tensors. Do not use for large sparse graphs (it
materializes `N x N` dense matrices and diagonal matrices), for learned or
batched adjacencies (it is numpy, not differentiable), or when a validated,
float32, ready-made support list is wanted (use `graph_utils`).

## Interface

All functions take `adj` as any array-like convertible to a square 2-D array.
They return a new dense `[N, N]` `float64` numpy array and are stateless.
`ValueError` is raised for non-2-D or non-square input. NaN/inf in `adj` are not
checked, and negative row sums are not handled (a negative degree gives NaN under
`D^{-1/2}`). Non-symmetric inputs are allowed and not symmetrized. The input is
never modified.

- `symmetric_normalized_laplacian(adj)`: degrees from row sums.
- `lambda_rescaled_laplacian(adj, lambda_max=2.0)`: `lambda_max` (float, nonzero; zero raises
  `ZeroDivisionError`) is
  supplied by the caller; the default 2 gives `L - I`. It does not compute
  the eigenvalue and does not symmetrize.
- `gcn_norm(adj)`: adds self-loops (identity) before degrees, so existing
  diagonal entries are increased by 1.
- `transition_matrix(adj)`: rows sum to 1 except zero-degree rows (all 0).
- `reverse_transition_matrix(adj)`: normalizes by the in-degree
  (column sums of `adj`).
- Private helpers `_as_dense`, `_inv_sqrt_degree`, `_inv_degree` are not public.
