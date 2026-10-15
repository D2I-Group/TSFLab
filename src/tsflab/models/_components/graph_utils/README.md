---
name: "graph_utils"
description: "Adjacency support builders: normalize_adj_mx modes (normlap, scalap, symadj, transition, doubletransition, identity, origin), torch conversion, and Chebyshev polynomials. Use for models taking a precomputed dataset graph; not for learned or adaptive graphs, GCN renormalization, or a user-set lambda_max."
---

# graph_utils

## What it does

Turns one adjacency matrix into the list of dense supports a graph forecaster
consumes. `normalize_adj_mx(adj, adj_type)` returns, per mode:

- `normlap`: `I - D^{-1/2} A D^{-1/2}`, degrees from the row sums of `A`, `A` not symmetrized, zero-degree rows give zeros in `D^{-1/2}`;
- `scalap`: scaled Laplacian `2L/lambda_max - I` from `graph_spectral.scaled_laplacian`, which first symmetrizes `A` as `max(A, A^T)` and uses the true largest `|eigenvalue|` of `L` (`L - I` if that is below 1e-12); unlike `normlap` it is therefore not the plain row-sum Laplacian of a directed `A`;
- `symadj`: `I - normlap = D^{-1/2} A D^{-1/2}` (same row-sum, unsymmetrized convention as `normlap`);
- `transition`: random walk `D^{-1} A` (zero-degree rows stay zero);
- `doubletransition`: `[D^{-1} A, D_r^{-1} A^T]` (forward and reverse walks, two supports);
- `identity`: `I`;
- `origin`: `A` with diagonal set to 1.

## When to use

Use when a model takes a precomputed adjacency and needs a standard list of
normalized supports. Do not use for learned/adaptive graphs (see
`adaptive_node_embedding_adjacency`), for GCN renormalization with self-loops
(`adj_norm.gcn_norm`, not exposed here), or when `lambda_max` must be user-specified
(`graph_spectral.scaled_laplacian` always computes the true value).

## Interface

Module-level functions (the catalog lists `normalize_adj_mx`, `adj_to_supports`, `cheb_poly`; no state). All
inputs are cast to float64 numpy internally; outputs are float32.

- `normalize_adj_mx(adj_mx, adj_type, return_type="dense") -> list`: `adj_mx` must be a
  finite square array (`ValueError` otherwise). `adj_type` in the modes above;
  unknown raises `ValueError("unknown adjacency normalization: ...")`.
  `return_type="dense"` gives float32 numpy arrays, `"coo"` gives
  `scipy.sparse.coo_matrix` with the same values; other values raise `ValueError`.
  One support for every mode except `doubletransition` (two). The adjacency is not
  symmetrized for `transition` modes; `scalap` symmetrizes (via `graph_spectral`).
- `adj_to_supports(adj_mx, adj_type="doubletransition", device="cpu") -> list[torch.Tensor]`:
  dense float32 tensors on `device`.
- `cheb_poly(matrix, order) -> np.ndarray`: thin wrapper of
  `graph_spectral.chebyshev_polynomials`; exactly `order` polynomials starting from
  identity, `ValueError` for `order < 1`.
