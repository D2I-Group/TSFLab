# adain_style_norm — reference

## Origin and granularity

The computation is the AdaIN operation of image style transfer (Huang and
Belongie, ICCV 2017), applied in CANet as its non-stationary adaptive
normalization (NSAN) after the spectral block, in place of a fixed LayerNorm
affine. It was added as a shared component in the CANet intake commit
(`b94ed873`, "add TQNet, TimePro, Gateformer, CANet"). The cut is only the
normalize-then-rescale arithmetic. Deriving the style statistics stays in
`canet`: its projection of the raw series mean/std, its `StyleBlendingGate`
(a fixed-ratio convex blend of two (mean, std) pairs followed by linear
projections, not a learned gate), and the style branch computed from patch
embeddings.

## Invariants and equivalence evidence

- A contract check written at extraction verified the empty state dict, output
  shape and dtype, that output mean and unbiased std over the length axis equal
  the style statistics (loose tolerance because of eps), that `x`, `style_mean`
  and `style_std` all receive gradients, scalar-style broadcasting, and the
  `ValueError` for a 2-D input; the seeded output and `x` gradient were pinned
  as a regression value.
- A model-level check in the pre-consolidation suite repeated the statistics
  check on a shifted, scaled input with constant style statistics.
- The detached `mu`/`sigma` claim is by construction; no check isolated it. No
  pre-refactor reference exists (the component was created together with its
  only consumer, `canet`), whose forward and backward were exercised by a
  `canet` forward/gradient test in the same pre-consolidation suite. These
  checks passed in the full suite run of 2026-10-03 before the test suite was
  consolidated.

## Variants and options

None beyond `eps`. It is a different operation from `revin` (which uses its own
statistics and restores them later with an explicit denorm) and from
`last_value_center` (level-only centering, no scale).

## Related components

`revin` (self-statistics, reversible), `last_value_center` (level-only
centering). The gate that blends the style statistics in `canet` is model-local
and is not `gated_fusion`: `gated_fusion` is a learned sigmoid gate over two
embeddings, whereas `canet` blends statistics with a fixed ratio.
