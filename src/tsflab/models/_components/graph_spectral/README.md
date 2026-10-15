---
name: "graph_spectral"
description: "Scaled normalized Laplacian (2L/lambda_max - I, safe for degenerate graphs) and exact-order Chebyshev polynomial stacks. Use for Chebyshev spatial convolutions on a fixed adjacency, including isolated nodes; not for random-walk or diffusion supports, learned or dynamic graphs, or very large N."
---

# graph_spectral

## What it does

Prepares supports for Chebyshev graph convolution. For adjacency `A`, degree
`D = rowsum(A)`, the normalized Laplacian is `L = I - D^{-1/2} A D^{-1/2}` (inverse
root of zero degree set to 0) and the scaled Laplacian is
`L~ = 2 L / lambda_max - I`, where `lambda_max` is the largest absolute eigenvalue of `L`
(if `lambda_max < 1e-12`, `L~ = L - I`; this is the case for an identity or
self-loop-only graph, where `L = 0` and `L~ = -I`). The Chebyshev stack is
`T_0 = I, T_1 = L~, T_k = 2 L~ T_{k-1} - T_{k-2}`.

## When to use

Use for Chebyshev-polynomial spatial convolutions over a fixed adjacency, especially
when the graph can contain isolated nodes or be an identity. Do not use for
random-walk or diffusion supports (see `graph_utils`), for learned or dynamic
graphs (the eigendecomposition is host-side NumPy at construction time, `O(N^3)`),
or for large `N` where a dense `[order, N, N]` stack is too big.

## Interface

All three are module-level functions (no `__all__`); no parameters or state.

- `scaled_laplacian(adj_mx, *, undirected=True) -> np.ndarray`: `adj_mx` is any array
  convertible to float64 `[N, N]`. Raises `ValueError` for non-square input or
  non-finite values. With `undirected=True` the adjacency is symmetrized with
  `max(A, A^T)` and `eigvalsh` is used; with `False`, row degrees of the directed matrix
  and general `eigvals` are used (a non-negligible imaginary part is discarded when casting to float32, with a
  NumPy `ComplexWarning`).
  `N = 0` returns an empty array. Returns float32.
- `chebyshev_polynomials(matrix, order) -> np.ndarray`: exactly `order` matrices
  `[order, N, N]` in the dtype of `matrix`, starting with identity. Raises
  `ValueError` for `order < 1` or non-square input.
- `chebyshev_supports(adj_mx, order, *, undirected=True) -> torch.Tensor`: composition of the two
  above, float32 tensor `[order, N, N]` on CPU. `order=1` returns only the identity.
  Note `order` counts polynomials, so a "K-th order" convention (`T_0..T_K`) needs `K + 1`.
