# series_decomposition — reference

## Origin and granularity

This is the `series_decomp` / `moving_avg` block of Autoformer, also used in
FEDformer and DLinear. It was extracted in commit `15bc27e6` ("extract series
decomposition") from near-identical local copies in `amplifier`, `bist`,
`moderntcn`, `stop`, and the shared `autoformer_encdec` and `dlinear`
components; `15bc27e6`'s follow-up `b1518394` migrated `timemixer`. The cut
is the moving average plus subtraction. Model-local: multi-kernel or
learned decompositions (for example MICN-style several kernels are just several
instances), DUET's per-expert moving average (kept local on purpose, because
it supports even kernels through asymmetric padding while this component
rejects them, which `bist` relies on), and what is done with the two parts.
Current consumers: `autoformer`, `fedformer`, `micn`, `symtime`, `refocus`,
`amplifier`, `stop`, `timemixer`, `moderntcn`, `bist`, plus the `dlinear`
component.

## Invariants and equivalence evidence

- Contract check (at extraction): the empty state dict, shape and dtype, `residual + trend == x`,
  the hand-computed first-step trend, a constant series unchanged, gradients, `kernel_size > length`,
  `stride=2` giving length 5 from 10, kernel 1, the `ValueError` cases (kernel 0, -3, 4;
  stride 0); a seeded output was pinned as a regression value.
- Reference comparison (repository contract check): output equals explicit edge padding plus
  `avg_pool1d`, `residual + trend == x`, gradients finite, even kernel
  rejected.
- Equation check (an `autoformer` model test): the hand-computed trend
  `[5/3, 3, 13/3]` for `[1, 3, 5]` with kernel 3.
- Equivalence of the migrated `timemixer` against pre-refactor values (outputs, state dict,
  gradients) was checked at extraction.
- An extraction check pinned that the shared component rejects even kernels while DUET's
  local helper accepts them.
- These checks, including the pre-refactor equivalence, passed in the full suite run of
  2026-10-03 before the test suite was consolidated.

## Variants and options

- `EdgePaddedMovingAverage` alone for smoothing only (`refocus`, `timemixer`).
- `stride > 1` downsamples the smoothed series; the decomposition class always
  uses stride 1.
- Not provided: even kernels, learnable kernels, zero or reflect padding, and
  decompositions by frequency (see `haar_dwt1d`, `wavelet`).

## Related components

`dlinear` (composes it with two projections), `haar_dwt1d` and `wavelet`
(frequency-based splits of the same trend/detail kind, not moving-average), `revin` and
`last_value_center` (other ways of removing level from a series before the model, with no
trend/residual pair).
- `freq_band_moe`: learned spectral band split instead of a moving-average trend.
- `decomposition_encdec`: Autoformer/FEDformer encoder and decoder layers that apply this decomposition after every sub-layer.
