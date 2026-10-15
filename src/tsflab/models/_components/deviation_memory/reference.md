# deviation_memory — reference

## Origin and granularity

Added with `st_ssdl` (commit `6663e2e0`, "add VisiFold, Extralonger, ST-SSDL,
RAGC", automated intake) as the paper-neutral part of its self-supervised deviation
mechanism; `st_ssdl` is the only consumer. Model-local: the Chebyshev graph-GRU
encoder, the triplet margin loss on (query, nearest, second_nearest), the L1
`deviation_loss` between latent and prototype deviations, and the pairing of a
current and a historical window. The component names no spatiotemporal axis:
`h` may carry node, batch or any leading shape.

## Invariants and equivalence evidence

- Contract checks verified the state-dict keys (`prototypes`, `query_proj`), the
  output types and shapes (indices are `long`, the two indices differ), that
  `indices`, `nearest` and `value` equal the Purpose formulas recomputed from
  `query` and `prototypes`, a 1-D `h` giving `[prototype_dim]` outputs, gradients
  reaching `h` and every parameter, the `ValueError` cases, and the L1-sum value
  `[6, 0]` of `deviation_score` with its zero-for-identical and gradient
  behavior. Seeded `value`, `query` and `indices` were pinned as regression
  values.
- `st_ssdl` model-level checks pinned the five output shapes for
  `h: [B, N, query_dim]`, `deviation_score(x, x) == 0` exactly and
  `deviation_score >= 0`, and verified that `st_ssdl`'s contrastive and
  deviation losses are finite and that the deviation loss is about 0 for an
  identical history and larger for a perturbed one.
- These checks passed in the full suite run of 2026-10-03 before the test suite
  was consolidated.

## Variants and options

Only the three sizes. No temperature on the softmax, no top-k other than 2, no
hard-assignment or EMA prototype update, no cosine scoring (the score is a raw dot
product of the projected query with prototypes).

## Related components

`periodic_query_bank` (also a learned bank, but indexed by integer phase with no
attention or nearest-prototype lookup; this one is queried by content),
`regularized_adaptive_graph_conv` (extracted in the same intake batch; no functional
overlap, only the shared consumer `st_ssdl`).
