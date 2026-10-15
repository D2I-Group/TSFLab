# wavelet — reference

## Origin and granularity

Created in commit `d73cf9d0` (automated intake of DPWMixer, AWEMixer, TimeExpert,
WDformer) as the shared wavelet layer. Consumers: `wdformer` (haar decimated,
`level=wave_size`, one instance to embed and one to reconstruct the forecast with
`reconstruct`), `dpwmixer` (haar `level=1`, keeping only the approximation to
build a downsampling pyramid), `awemixer` (undecimated; db4 level 3 are the defaults in
both the component and the model; subbands feed an adaptive router).
The cut is the filter bank and its inverse; learned mixing, routing, and
per-subband heads stay in the models. The single-level stateless Haar pair is a
separate component, `haar_dwt1d` (from `swift`). `WPMixer` and `sonnet`
define wavelet code locally and do not use this component.

## Invariants and equivalence evidence

- Contract checks (at extraction): wavelet names and `ValueError` cases; the decimated
  haar round trip (level 2 at `L=16` and odd `L=15`, state-dict keys `low`/`high`);
  decimated db4 shapes and no-reconstruct (db4 subband shapes `[9, 9, 11]`,
  `NotImplementedError`, wrong coefficient count); orthonormal filters (unit norm and
  orthogonality of haar/db2/db4); the undecimated short-input error; and undecimated
  shapes and gradient (equal-length subbands, dtype, input gradient, seeded reference
  values pinned as a regression value).
- Consumers: `wdformer`, `dpwmixer` (a haar level-1 decomposition of `[2, 4, 12]`
  gives two `[2, 4, 6]` bands) and `awemixer` (db2 level 2) structure tests exercised
  the transforms through their models; none checked wavelet numerics against `pywt`,
  and nothing compared against a reference wavelet package. These checks passed in
  the full suite run of 2026-10-03 before the test suite was consolidated.

## Subband lengths

Haar lengths follow `ceil(L / 2)` per level. Longer filters give longer
subbands: per level `floor((L' + 2*(filter_len - 2) - filter_len) / 2) + 1` with
`L'` the even padded length; db4 at `L=16` level 1 gives 11 samples, not 8, and
the contract check pinned `[9, 9, 11]` for level 2.

## Stateful reconstruction

`DecimatedWaveletTransform.reconstruct` defaults to the odd-padding flags
(`last_trims`) recorded by the most recent `decompose` call. Round trips are
checked at `L=16` and `L=15`, level 2, with the default and (in a component
validation check) with explicit `trims` after the recorded
state was overwritten. Alternating different lengths on one instance corrupts the
default trims, which is why `wdformer` keeps separate instances.

## Variants and options

`wavelet` in `available_wavelets()`; `level`; decimated vs undecimated class.
Exact inverse exists only for haar/db1 decimated. For one level, stateless Haar
use `haar_dwt1d`.

## Related components

- `haar_dwt1d`: stateless one-level Haar pair with an exact inverse for any length;
  use it for a single level (this module's haar reconstruct is stateful and its
  odd-length padding is replicate, versus the length argument of the Haar pair).
- `series_decomposition`: moving-average trend/seasonal split in the time domain.
- `freq_band_moe`, `frequency_band_sampler`: frequency-domain (rfft) band
  decompositions; learned or depth-indexed instead of fixed wavelet filters.
- `fft_extrapolation_conv`, `spectral_descriptor`: other spectral-domain tools; no
  wavelet subbands.
