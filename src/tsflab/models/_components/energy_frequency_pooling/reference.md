# energy_frequency_pooling — reference

## Origin and granularity

Extracted from `refocus` (commit `6b491e13`, automated intake of ReFocus,
FreqMoE, SWIFT, Sensorformer), which uses it inside its key-frequency block:
rfft of hidden features, pooling, irfft, then fusion with the original input.
The module docstring says the deterministic eval argmax is a documented
train/eval difference, as the paper only describes the stochastic training
behaviour. Cut at the pooling operator; the FFT/IFFT pair, the fusion layers,
and the choice of what the token axis is stay in `refocus`.

## Invariants and equivalence evidence

- A model-level check verified, in eval, that both tokens receive the
  higher-energy token's value.
- Contract checks verified in eval that the output keeps shape and complex
  dtype, has an empty state dict, is identical across tokens and equals the
  argmax-energy token per bin; in train that every pooled value equals one of
  the token values and gradients reach the input; and both `ValueError` cases.
  The seeded eval output was pinned as a regression value. These checks passed
  in the full suite run of 2026-10-03 before the test suite was consolidated.
  The training-mode sampling distribution itself was not checked statistically.

## Variants and options

None. There is no temperature, top-k, or straight-through variant; making
eval stochastic or the softmax temperature configurable would be a new
component.

## Related components

`harmonic_energy_gate` (also energy-driven, but gates harmonic bins of a series rather than
selecting one token per bin), `freq_band_moe` (splits one series into frequency bands and
softmax-mixes them; no cross-token selection), `frequency_band_sampler` (slices a fixed
bin range; no energy). `revin` is only the normalization `refocus` applies before this block.
- `spectral_descriptor`: window-level entropy and band ratios rather than token selection.
