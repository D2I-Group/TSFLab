# frequency_band_sampler — reference

## Origin and granularity

Extracted from `dualformer` (commit `dd63af6c`, automated intake of SDMixer,
SEMixer, LSINet, Dualformer), where the frequency branch gives each encoder
layer its own band. The docstring describes it as the band arithmetic of
several dual time-frequency architectures; only Dualformer is recorded as the
source. Model-local in `dualformer`: the rfft/irfft around it (it builds a
band-limited signal by zeroing bins outside the range), the encoder layers, and
the fusion with the time branch.

## Invariants and equivalence evidence

- Numeric-fix check (pre-consolidation suite, which passed in full on 2026-10-03):
  exact tiling (union equals every bin, adjacent bands meet, layer 0 ends at `n`)
  over many bin and layer counts, and that sliding windows reach both ends (`alpha`
  in 0.3, 0.55, 1.0), plus the `alpha=1.0` full-width case. The `n < num_layers`
  fallback, argument validation, and `sample` were not directly checked.
- A `dualformer` model check (pre-consolidation suite) confirmed distinct per-layer
  bands with layer 0 at higher frequency than the last layer.
- No numerical reference tensor: this is integer band arithmetic.

## Variants and options

`alpha` switches between tiling (`alpha <= 1/num_layers`) and overlapping
sliding windows (larger `alpha`). There is no log-spaced or learned variant.

## Related components

`freq_band_moe` (also splits an rfft into contiguous bands, but with learned
boundaries and gated mixing inside one block, whereas this one is fixed
arithmetic that assigns one band per depth), `harmonic_energy_gate` (the other
Dualformer frequency component), `energy_frequency_pooling` (energy-driven
selection over tokens rather than fixed bins), `wavelet` (multi-resolution
alternative).
- `dominant_periods`: period selection rather than band sampling.
