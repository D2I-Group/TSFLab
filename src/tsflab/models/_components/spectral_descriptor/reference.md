# spectral_descriptor — reference

## Origin and granularity

Extracted from `core` (CoRe). CoRe feeds the descriptor
into a gate that scales its cross-variate correction. The cut is at the
statistic itself: the gate network, the correction branch, and how the gate is
applied stay in `core`. The existing `harmonic_energy_gate` and
`frequency_band_sampler` were retrieved first and rejected: they return a
per-channel periodicity ratio and sampled band indices, not window-level
entropy plus band ratios.

## Invariants and equivalence evidence

- Reference comparison (a `core` test-time-adaptation test): the module was compared
  with an independent float64 oracle over several lengths, with checks of the entropy
  range and that the band ratios sum to one.
- Differences from the official CoRe code are recorded in the `core` card:
  statistics are per sample (official: per batch) and band edges follow the
  paper's inclusive ranges (official: half-open thirds).
- A contract check (at extraction) pinned the interface (empty state dict, shape and
  dtype, entropy range, ratios summing to one, flat-spectrum entropy near 1 for constant
  input, a low tone landing in the low band and a high tone in the high band, errors and
  gradient flow) and a seeded numerical regression value.
- These checks passed in the full suite run of 2026-10-03 before the test suite was
  consolidated.

## Variants and options

`eps` only. Band edges are fixed thirds of the one-sided spectrum; a different
partition belongs in a new option, not in a consumer-side edit.

## Related components

`harmonic_energy_gate` (per-channel periodicity ratio from harmonic bins),
`frequency_band_sampler` (depth-indexed band bin ranges, no statistics),
`dominant_periods` (explicit top-k period discovery rather than a distribution
summary), `energy_frequency_pooling` (energy-driven bin selection across tokens).
