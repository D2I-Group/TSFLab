# freq_band_moe — reference

## Origin and granularity

Extracted from `freqmoe` (commit `6b491e13`, automated intake). The module
docstring calls it a paper-neutral block for frequency-domain MoE forecasters.
Kept here: normalization, band split, gate, recombination, and restoration of
the instance scale. Model-local in `freqmoe`: the downstream frequency-extension
blocks (complex linear layers upsampling the spectrum from `seq_len` to
`seq_len + pred_len`), the complex ReLU/dropout, and the use of the returned
boundaries and gate scores (stored as `last_band_boundaries`, `last_gating_scores`).

## Invariants and equivalence evidence

- A structural check covered output shape, boundary vector length and
  monotonicity, and gate rows summing to 1.
- No pre-refactor tensor reference exists; consumer-level behaviour was covered by
  `freqmoe` model checks (pre-consolidation suite).
- Numeric-fix checks: the default `band_boundaries` is a buffer with no gradient,
  round-trips through `state_dict`, and a learnable module's checkpoint loads into
  the default one; with learnable boundaries the forward output, boundaries and
  gates equal the fixed module's and `band_boundaries.grad` is finite and nonzero;
  `expert_num == 1` returns `[0.0, 1.0]` with finite output. These passed in the
  full suite run of 2026-10-03 before the test suite was consolidated. Nothing
  compares the default forward against the pre-change implementation; that was a
  one-off manual check and is not reproducible from the repository.
- Paper vs code: the paper (Sec. on the frequency-decomposition MoE) says the
  boundaries are learned end-to-end, but the official code casts them to integers,
  which blocks every gradient. The default follows the official code (fixed,
  random-initialization boundaries; a buffer, so the module no longer advertises a
  parameter that cannot train). Learning is opt-in.
- Bands that round to zero width are empty (their mask is all zero). With
  `expert_num == 1` the single band covers all bins and `boundaries == [0, 1]`.

## Variants and options

`learnable_boundaries=True` turns `band_boundaries` into a Parameter trained with
a straight-through estimator: the forward uses the exact hard 0/1 band masks, the
backward uses soft masks `sigmoid((bin + 0.5 - edge) / T)` differences, with
`edge = boundary * freq_len`. `boundary_temperature` is `T` (bins). This is not in
the paper or official code. Boundaries are shared over channels and batch;
only the gate depends on the input, and it sees the channel-averaged amplitude
spectrum.

## Related components

- `frequency_band_sampler`: deterministic depth-indexed bands over an FFT axis;
  no gate, no learned boundaries, no recombination.
- `revin`: external reversible normalization; this module normalizes and restores
  scale internally with its own (non-affine) statistics.
- `series_decomposition`: time-domain trend/residual split; here the split is
  spectral and recombined by gates.
- `topk_expert_router`: input-conditioned routing over experts that is not tied to
  a spectrum; this module's gate is dense softmax (no top-k).
- `wavelet`, `haar_dwt1d`: alternative multi-band decompositions (fixed filter
  banks) instead of learned rfft band edges.
