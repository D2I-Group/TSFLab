# channel_alignment — reference

## Origin and granularity

Extracted in "extract reusable forecast contracts" (`db5b2970`) to replace
repeated model-local `_fit_channels` helpers; at extraction a repository
contract check forbade reintroducing a local `_fit_channels`. Current consumers are the spatiotemporal recurrent models
`dcrnn`, `gclstm` and `gts`, which pass `fit_channels(to_spatiotemporal(x_enc,
x_mark_enc), input_dim)` to align the (value + mark) features with a fixed cell
input width. The original per-model helpers' exact origin is not recorded in
history. What stays local: choosing `width`, building the marks, and the model
that consumes the aligned tensor.

## Invariants and equivalence evidence

- A contract check verified the pad and slice shapes, that padding is zero and
  the kept channels are unchanged, the `C == width` case (new object, equal
  values), gradient flow (`[2, 2, 1]` per channel for one pad plus one slice),
  float64 and int64 dtype preservation, and the `width=0` error; seeded
  reference values were pinned as a regression value.
- A repository contract check verified the slice case, the zero-padded case
  (leading channels preserved, padding is zeros), and the `width=0` error. Both
  checks passed in the full suite run of 2026-10-03 before the test suite was
  consolidated.
- No pre-refactor model reference: the extraction from the three consumers has
  no stored before/after outputs; the consumers were covered only by the generic
  model contract checks.

## Variants and options

None. Left-padding, learned projections, or reflect padding are not provided.

## Related components

- `marks`: `to_spatiotemporal` builds the tensor typically aligned here.
- `forecast_embedding`: maps value channels plus calendar marks to `d_model` by a
  learned projection; use it instead when information must not be dropped.
- `channel_wise_linear`: learned projection over the time axis, not the feature axis.
