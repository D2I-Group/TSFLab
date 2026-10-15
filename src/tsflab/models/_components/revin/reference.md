# revin — reference

## Origin and granularity

The block is the standard RevIN layer used by many forecasters; it was
consolidated from near-identical model-local copies (commit `61451843`,
"consolidate reversible normalization", replaced local copies in crib, glocalib,
mgsfformer, mofo, pathformer, timealign, timefilter, timekan, and timemixer) and the
`patchtst` backbone. It is cut at the layer boundary: normalization statistics,
affine parameters, and inversion. Model-specific uses stay local: where `norm`
and `denorm` are called, forecast-length handling, other normalizations
(per-patch, non-stationary de-stationary attention factors), and last-value
variants that do not store state (use `last_value_center` for those).

## Invariants and equivalence evidence

- Contract check (at extraction; pre-consolidation suite), parametrized over
  `affine`/`subtract_last` = (T,F), (F,F), (T,T): state-dict keys, zero mean and
  unit variance when neither option is set, a zero last step for `subtract_last`,
  the `denorm(norm(x))` round trip, a different horizon length at `denorm`,
  gradient to the input and affine parameters, and the rank-4 shape; a companion
  check covered the `RuntimeError` and `ValueError` cases and the identity of the
  disabled module. Seeded reference values for (affine, mean), (no affine) and
  (affine, last-value) were pinned as regression values.
- Round trip: `denorm(norm(x)) == x` up to float error for the default,
  `subtract_last=True`, and disabled configurations; checked by a repository
  contract test (pre-consolidation suite).
- Disabled mode returns the identical tensor object (same check).
- The graph-component frozen-copy extraction test constructed RevIN inside a
  graph-consumer reference model; no dedicated pre-refactor fixture exists for
  RevIN (consolidation was checked through the consumers' contract tests). These
  checks passed in the full suite run of 2026-10-03 before the test suite was
  consolidated.
- State-dict keys are `affine_weight` and `affine_bias` only, so checkpoints of
  models that held a local copy keep loading.

## Higher-rank inputs

Mean, variance, and the `subtract_last=True` center all reduce over the same
axes, so each has shape `[B, 1, ..., 1, C]`. For rank > 3 the `subtract_last`
center is the last step along axis 1 averaged over the extra axes `2..ndim-2`;
for rank 3 there are no extra axes and it is exactly `x[:, -1:]` (unchanged from
before this fix, which only altered rank > 3). All model consumers pass rank-3
`[B, L, C]` histories, so their behavior is unchanged;
a component validation check (pre-consolidation suite) pinned the rank-4 shapes.

## Variants and options

- `affine=False` for pure statistics normalization.
- `subtract_last=True` for the last-value-centred variant: the level is the
  last observed step, the scale is still the standard deviation.
- `enabled=False` to ablate normalization without changing the call sites.
- Materially different normalizers (series stationarization with learned
  de-stationary factors, per-patch normalization, AdaIN with external statistics)
  are separate: see `adain_style_norm` and `last_value_center`.

## Related components

- `last_value_center`: stateless last-value centering, no scaling; the caller keeps
  the level instead of the module.
- `adain_style_norm`: rescales to externally supplied statistics.
- `series_decomposition`: often applied after normalization (trend/residual split).
- `patchtst`: backbone that composes RevIN.
- `freq_band_moe`: a block that instance-normalizes internally (non-affine, not
  invertible by the caller), so do not stack it with RevIN unintentionally.
