# harmonic_energy_gate — reference

## Origin and granularity

Introduced with `dualformer` as a standalone module (commit `dd63af6c`, automated
intake of SDMixer, SEMixer, LSINet and Dualformer). The
consumer applies it to the embedded sequence `[B, L, d_model]` and mixes
`freq_state * w + time_state * (1 - w)`; the code comment calls this the paper's
periodicity-aware gate. Cut at the ratio itself; the two branches, the fusion
rule, and the choice of what to feed it remain in `dualformer`. The module is
parameter-free.

## Invariants and equivalence evidence

- Contract check (at extraction): an empty state dict, the `[2, 1, 3]` shape and
  dtype, outputs in `[0, 1 + 1e-6]`, a pure 4-cycle sinusoid over length 64 scoring above
  0.99, white noise scoring below it, and seeded values
  pinned as a regression value; the `ValueError`s (non-positive constructor
  arguments, rank not 3, too-short sequence); float64 output and finite gradients.
  These checks passed in the full suite run of 2026-10-03 before the test suite was consolidated.
- The ratio never exceeds 1 up to the `1e-5` stabilizer: the fundamental is below
  `nb // num_harmonics`, so `num_harmonics * f0 <= nb - 1`, the `clamp` never fires, and
  each harmonic bin is distinct and counted once.
- A pure sinusoid scores just under 1 (`E / (E + 1e-5)`), not exactly 1.
- `dualformer` model checks (pre-consolidation suite) exercised the gate end to end.

## Variants and options

`num_harmonics` and `low_freq_guard` only. The ratio uses the sample's own
spectrum (no running statistics), so it is recomputed per forward call.

## Related components

`dominant_periods` (explicit period discovery), `gated_fusion` (learned fusion
gates; this component is parameter-free and spectrum-driven), `frequency_band_sampler`
(the other Dualformer frequency component), `spectral_descriptor` (per-window spectral
entropy and band-energy ratios, a different spectral summary), `energy_frequency_pooling`.
