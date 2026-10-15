# graph_spectral — reference

## Origin and granularity

Introduced in commit `1822ab5d` ("rewrite(models): verify graph ssm and physical
families"), a large commit that rewrote graph models (including `astgcn` and `gclstm`)
away from vendored upstream code. The commit message gives no rationale for this
file; the "degenerate" keyword and the test name indicate it was written to handle
isolated nodes and identity graphs. Which upstream code it replaced is not recorded.
Current importers are `astgcn`, `stgcn`, `gclstm`, and `dstagnn` (all call
`chebyshev_supports`), and `graph_utils` reuses `scaled_laplacian` and
`chebyshev_polynomials`. The boundary is the numeric support
construction only. The Chebyshev convolution layer, the choice of order `K`, where
supports are registered as buffers, and the temporal blocks stay model-local. Note a
second, simpler `lambda_rescaled_laplacian(adj, lambda_max=2.0)` exists in `adj_norm`; the two
differ (this one computes the true `lambda_max`, symmetrizes, and rejects non-finite input).

## Invariants and equivalence evidence

- Degenerate-graph check: identity graph gives a
  finite scaled Laplacian, `chebyshev_polynomials(.., 1)` has shape `(1, 3, 3)`,
  `chebyshev_supports(.., 3)` has shape `(3, 3, 3)` and is finite, `order=0` raises.
- Contract check (at extraction): float32 symmetric
  output for undirected input with spectrum in `[-1, 1]`, directed mode shape/dtype,
  zero and `0 x 0` adjacency, `ValueError` for non-square and NaN input, the
  recurrence `T_2 = 2 L~^2 - I`, `T_3 = 2 L~ T_2 - T_1`, `order=1` and `order=0` behaviour,
  and seeded output values pinned as a regression value. These checks passed in
  the full suite run of 2026-10-03 before the test suite was consolidated.
- No pre-extraction reference from the consumer models; consumer-level behaviour is
  covered by those models' own checks.

## Variants and options

- `undirected=True` (default) symmetrizes; `undirected=False` keeps direction.
- `order` is exact (not `K + 1`); the identity is always the first support.
- Supports are dense; sparse graphs are densified.

## Related components

`graph_utils` (list-based support API, `scalap` mode wraps this), `adj_norm` (cheaper
fixed-`lambda_max` rescaling `lambda_rescaled_laplacian`, non-symmetrizing and not
rejecting non-finite input; use this component when the true `lambda_max` matters), `diffusion_conv`
(applies supports as repeated powers), `adaptive_node_embedding_adjacency` (learned
adjacency to feed in).
