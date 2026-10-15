# adj_norm — reference

## Origin and granularity

The math is the published standard; the module docstring records that the
definitions are also used by BasicTS `adjacent_matrix_norm.py` (Apache-2.0). It
began as an external utility module and now sits among the shared components. No
model package imports it directly and no origin model is recorded; the component
audit exempts it from the "has a consumer" rule for that reason. It has two
consumers. The `graph_utils` component uses `symmetric_normalized_laplacian`,
`transition_matrix`, and `reverse_transition_matrix` to build the named supports
`normlap`, `symadj` (`I - L`), `transition`, and `doubletransition`.
`gcn_norm` and `lambda_rescaled_laplacian` are not used by `graph_utils`. The
experiment runner (`src/tsflab/experiments/runner/run_one.py`, `_normalize_adj`)
applies any of the five functions to a data-derived adjacency when the dataset
parameter `adj_norm` is set. Input validation of finite values, the choice among
normalization types, float32 and torch conversion, and the eigenvalue-based scaled
Laplacian stay in `graph_utils` / `graph_spectral`.

## Invariants and equivalence evidence

- A contract check verified that all five outputs are `[N, N]` `float64` and
  finite, transition rows sum to 1 except zero-degree rows (0),
  `reverse_transition_matrix(A) == transition_matrix(A.T)`,
  `lambda_rescaled_laplacian` equals `L - I` at the default and `0.5 L - I` at
  `lambda_max=4`, the Laplacian of a symmetric graph is symmetric, `gcn_norm` of
  an all-zero graph is the identity, and non-square input raises `ValueError`;
  seeded reference values were pinned as a regression value.
- A repository contract check verified that `gcn_norm` and `transition_matrix`
  are finite on a graph with an isolated node, and that `adj_to_supports`
  returns `transition_matrix(A)` and `transition_matrix(A.T)` as float32.
- Both checks passed in the full suite run of 2026-10-03 before the test suite
  was consolidated. No pre-refactor reference exists, so the `graph_utils`
  supports used by graph models (`dcrnn`, `gwnet`, `d2stgnn`, `dfdgcn`,
  `st_ssdl`) are covered only through those models' own contract checks.

## Variants and options

- `lambda_rescaled_laplacian(adj, lambda_max)` (formerly `scaled_laplacian`, renamed to
  avoid clashing with `graph_spectral.scaled_laplacian`) is a different function: that one symmetrizes by default,
  rejects non-finite input, computes `lambda_max` from the spectrum, returns
  float32, and is what `graph_utils` uses for `scalap`. This one only rescales
  with the given `lambda_max`.

## Related components

`graph_utils` (the consumer that builds supports), `graph_spectral`
(eigenvalue-scaled Laplacian and Chebyshev supports), `diffusion_conv` (consumes
the resulting supports),
`adaptive_node_embedding_adjacency` (learned adjacency, already normalized by
softmax).
