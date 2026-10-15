# weight_set_router — reference

## Origin and granularity

Extracted from the DiPE-Linear implementation, where the same routing mixes the
frequency attention, temporal attention and the frequential mapping. The
temperature schedule (annealing) is deliberately not part of the component; the
caller decides `tau`.

## Invariants and equivalence evidence

- Router checks (at extraction, in the `dipelinear` tests): column sums of one,
  the high-temperature uniform limit, that low temperature sharpens, `ValueError` at
  temperature 0, equality of `mix_weight_sets` with the explicit weighted sum and
  nonzero gradients; seeded reference values were pinned as regression values.
- Reference comparison: a `dipelinear` test loaded the official DiPE-Linear
  forecasts with the router logits copied in, so the routed `softmax(R / tau)` mix
  was checked end to end through `dipelinear`. These checks passed in the full suite
  run of 2026-10-03 before the test suite was consolidated. The component checks
  did not cover invalid sizes or a non-2-D routing.

## Variants and options

None. A hard assignment is the `tau -> 0` limit; a uniform average the `tau -> inf`
limit. Input-dependent routers belong in `topk_expert_router`.

## Related components

`fft_extrapolation_conv` (consumes the routing as `mixing`), `topk_expert_router`
(input-dependent, sparse expert gating; this router is a static learned matrix),
`sparse_connection_router` (learned binary connections over positions, not a convex
mix of weight sets), `channel_wise_linear` (its `individual=False` and `True` modes are
the fully shared and fully per-channel ends of the range that this router interpolates
between).
