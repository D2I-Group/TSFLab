# diffusion_conv — reference

## Origin and granularity

Extracted in commit `2315e4e1` ("refactor(components): extract diffusion convolution")
from the `gwnet` and `sttn` upstream wrappers (the commit touched both
`_upstream.py` files); today `gwnet` and `stdmae` import it (`sttn` no longer does). The cut covers the
neighbourhood einsum, the 1x1 projection and the hop concatenation. The commit
message is empty, so the reason for the boundary is inferred from the code. Kept model-local: building the supports (`graph_utils`,
`adaptive_node_embedding_adjacency`), the gated dilated temporal convolution, the
residual and skip wiring, and the choice of `support_len` and `order`.

## Invariants and equivalence evidence

- A contract check verified `NeighborhoodConv2d` against the einsum and its empty
  state, the `PointwiseProjection` keys, the `[4, 15, 1, 1]` projection weight,
  output shape and dtype, eval determinism, gradients to the input and every
  parameter, the `ValueError`s for a wrong support count and for
  `support_len=0` or `order=0`, dropout active in training mode, and seeded
  values pinned as a regression value.
- A repository contract check compared the output against the explicit expansion
  `[x, S1 x, S1^2 x, S2 x, S2^2 x]` followed by the projection, plus finite
  gradients, and checked catalog registration.
- Equivalence of the `gwnet` model (which uses `DiffusionConv2d` with
  `support_len=3, order=2`) against a frozen verbatim pre-extraction copy was
  checked at extraction: identical state-dict keys, eval outputs and gradients.
  That frozen-copy test passed in the full suite run of 2026-10-03 before the
  test suite was consolidated. `stdmae` adopted the component later and has no
  frozen pre-extraction copy; `sttn` is not covered.

## Variants and options

- `order` controls hop depth; hop terms are repeated powers of the same support, not
  Chebyshev polynomials.
- `support_len` can be 1 for a single graph. Adaptive supports can be mixed with
  static ones in the list. Powers of the same support are recomputed per call.
- Dropout is applied after the projection, not on the hop terms.

## Related components

`graph_utils` (builds the supports), `adj_norm` (transition-matrix normalisers behind them),
`adaptive_node_embedding_adjacency` (adaptive support), `gated_dilated_conv` (the temporal half of a Graph WaveNet layer),
`graph_spectral` (Chebyshev supports), `regularized_adaptive_graph_conv` (linear-time alternative).
