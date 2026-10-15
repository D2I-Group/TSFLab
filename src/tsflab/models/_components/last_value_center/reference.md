# last_value_center — reference

## Origin and granularity

The scheme is NLinear's last-value subtraction. The helper pair was added in
commit `b1518394` ("last_value_center; migrate TimeMixer to
series_decomposition"); the commit message names the consumers NLinear,
SegRNN and CrossGNN (its anti-OOD branch), and states that CATS, HL and
PatchTST keep their own variants with documented divergences. Later consumers:
`mtlinear` (its `NLinear` layer type) and `composed` (the `last_value_center`
normalization slot, via the `LastValueNorm` adapter in `models/_slots/adapters.py`),
which are not in `origin_models` because they were not extracted from. It is cut at the
arithmetic only: where the call goes, the head, padding, and the choice to
bypass (CrossGNN passes `0.0` instead when `anti_ood=False`) stay local. The
state is explicit: the caller keeps `level`, so the pair is stateless,
unlike `revin`.

## Invariants and equivalence evidence

- Contract check (at extraction): shapes, that `level` does not require grad,
  that the last centered step is zero, that restoring the centered tensor
  recovers `x`, that a float `0.0` level is accepted, and that the input gradient
  is exactly ones (no path through the level); seeded reference values were pinned
  as a regression value.
- Equivalence against frozen pre-refactor outputs, state dicts, and input
  gradients for `nlinear`, `segrnn`, and `crossgnn` (with `anti_ood` true and
  false) was checked at extraction: identical state-dict keys and values, plus
  outputs and input gradients within `atol=1e-6`. That test passed in the full
  suite run of 2026-10-03 before the test suite was consolidated.
- An `nlinear` model-level check (pre-consolidation suite) showed a zero head
  returns the last observation repeated over the horizon.

## Variants and options

None. For mean/std normalization with reversible statistics use `revin`
(`subtract_last=True` gives a stateful last-value-centred variant that still
divides by the standard deviation). Not extracted: CATS keeps its own non-detached
variant (see the `cats` card), HL has no head between subtract and restore (see the
`hl` card), and PatchTST uses `revin(subtract_last=True)`.

## Related components

- `revin`: stateful mean/std alternative, optionally last-value-centred; also
  normalizes scale.
- `channel_wise_linear`: the NLinear head typically placed between centre and restore.
- `series_decomposition`: removes a smoothed trend, not just the last level.
- `adain_style_norm`: rescales to externally supplied statistics rather than the last step.
- `dlinear`: the NLinear/DLinear alternative that uses this centering.
