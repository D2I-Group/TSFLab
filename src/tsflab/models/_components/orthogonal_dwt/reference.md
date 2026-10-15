# orthogonal_dwt — reference

## Origin and granularity

The filter-bank DWT is Mallat's multiresolution algorithm (IEEE TPAMI 1989) with
Daubechies' compactly supported orthonormal filters (CPAM 1988); Symlets and
Coiflets are the near-symmetric families of Daubechies' *Ten Lectures on
Wavelets* (1992). The component consolidates two model-local implementations of
the same operator:

- `karma` (KARMA, arXiv 2506.08939, WASA 2025): single-level db4 with half-sample
  symmetric extension (`pytorch_wavelets` `mode="symmetric"`) along the embedding
  axis of every token, filters not checkpointed (`persistent=False`).
- `waverora` (WaveRoRA, arXiv 2410.22649): `J`-level sym3/coif3/haar with zero
  extension (`mode="zero"`), filters checkpointed as `dwt.analysis` and
  `dwt.synthesis`.

The cut is the filter bank, the boundary extension, and the multi-level
iteration. What the models do with the bands (Mambas on the coefficients,
per-band embeddings and predictors) stays local.

## Invariants and equivalence evidence

- Component checks (at extraction): orthonormality and the quadrature-mirror
  relations of every filter, coefficient lengths, the Haar closed form, perfect
  reconstruction in both modes (even and odd lengths, several levels), agreement
  with `pywt` (`dwt`, `idwt`, `wavedec`, `waverec`) in both modes when PyWavelets is
  installed, persistence of the buffers, the error contracts, and gradient flow.
- Migration of `waverora` (was `ZeroModeDWT`): the analysis padding
  `(L - 2, 2 floor((N + L - 1) / 2) - N)`, the stacked stride-2 correlation, the
  separate transposed convolutions with `padding = L - 2`, the odd-length trim, the
  buffer names `analysis`/`synthesis`, their registration order, persistence and
  values are unchanged, so outputs and state-dict keys are identical by
  construction; the API now works on any leading shape instead of reshaping
  `[B, M, N]` itself.
- Migration of `karma` (was `SymmetricDB4`): analysis is the same symmetric
  extension (`L - 1` mirrored samples each side, first sample dropped) and the
  same correlation with the same filter values, so it is bit-identical. Synthesis
  previously ran one transposed convolution over the stacked bands and cropped
  `[L - 2, L - 2 + 2n - L + 2)`; it now sums two transposed convolutions with
  `padding = L - 2`, which is the same linear map but sums in a different order
  (float32 differences of about one ulp). The filter buffer was non-persistent and
  stays so (no state-dict keys).
- At extraction, frozen copies of `SymmetricDB4` and `ZeroModeDWT` were compared
  under a fixed seed: identical state-dict keys of the migrated `karma` and
  `waverora` models, and equal outputs and parameter gradients (`atol = 1e-6` for
  outputs, `1e-5` for gradients) of the old and new transforms and models. That
  frozen-copy test passed in the full suite run of 2026-10-03 before the test suite was consolidated.

## Variants and options

- `mode`: `"zero"` (WaveRoRA) or `"symmetric"` (KARMA). Other PyWavelets modes
  (periodization, reflect, constant, periodic) are not implemented.
- `persistent`: `True` keeps the filter buffers in checkpoints (WaveRoRA);
  `False` keeps them out (KARMA). It does not change any output.
- Material variants kept elsewhere, not served by this component:
  `wavelet.DecimatedWaveletTransform` correlates with the unreversed filters,
  pads circularly and replicates an odd sample (different coefficients and
  alignment); `wdan`'s local `WaveletTrend` correlates with the unreversed
  `dec_lo` after a `L - 1`-sample symmetric extension (the filter mirrored and
  shifted by one sample relative to this convention); `db2_transf` learns its taps;
  `prism` builds same-length Haar bands with reflect padding and
  `hi = (odd - even) / sqrt(2)`; `haar_dwt1d` is the stateless one-level Haar pair
  with replicate padding.
- The db4 taps are the PyWavelets values; the `wavelet` component's db4 table
  agrees to about `1e-12` but is a separate table and is not shared.

## Related components

`wavelet` (circular decimated and a-trous transforms with its own alignment),
`haar_dwt1d` (stateless one-level Haar), `series_decomposition` (moving-average
trend split), `revin` (applied before the DWT in `waverora`).
