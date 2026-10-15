---
name: "wavelet"
description: "Fixed-filter wavelet transforms on [B, C, L]: multi-level decimated DWT (haar/db1/db2/db4, exact inverse for haar) and a-trous undecimated transform. Use for wavelet multi-scale forecasters (WDformer, DPWMixer, AWEMixer); not for pywt-exact coefficients or db2/db4 reconstruction."
---

# wavelet

## What it does

Two channel-independent (grouped, per-channel) wavelet analysis modules over
`[B, C, L]` with fixed filters (`haar`, `db1`, `db2`, `db4`):

- `DecimatedWaveletTransform.decompose`: stride-2 filtering per level on the
  running approximation, returning `[approx_L, detail_L, ..., detail_1]`
  (coarsest first, the `pywt.wavedec` order; the boundary handling is not
  pywt's, so coefficients are not numerically interchangeable with `pywt`).
- `UndecimatedWaveletTransform`: a-trous stationary transform, dilation `2**i`
  at level `i`, stride 1, so every subband has length `L`; returns the same
  coarsest-first list `[approx_L, detail_L, ..., detail_1]`.

The filter taps are entered as constants (closed form for haar/db2, tabulated
for db4), not imported from a wavelet package.

## When to use

Use when a model treats coarse and fine scales of each channel separately:
decimated for multi-level pyramids (invertible with haar), undecimated when each
subband must stay aligned per time step (no inverse). Do not use where
PyWavelets-exact coefficients (see `orthogonal_dwt`) or db2/db4 reconstruction
are required, where the length is shorter than the undecimated receptive field,
or when one decomposition instance is called on varying lengths before
`reconstruct`.

## Interface

`available_wavelets() -> tuple[str, ...]` returns the sorted names
`('db1', 'db2', 'db4', 'haar')`.

`DecimatedWaveletTransform(wavelet: str = "haar", level: int = 1)`

- `wavelet` must be in `available_wavelets()` (else `ValueError`); `level >= 1`
  (else `ValueError`).
- Buffers `low`, `high`, each `[1, 1, filter_len]`, float32 (they appear in the
  state dict and follow `.to(device/dtype)`); inputs of another dtype need the
  module cast with `.to(dtype)` first. No parameters. `x` must be 3-D (unpacked as
  `batch, channels, length`); other ranks raise `ValueError` from unpacking.
- `decompose(x)`: odd lengths are replicate-padded by one sample per level;
  filters longer than 2 taps also pad circularly by `filter_len - 2` each side,
  so their subbands are longer than `ceil(L / 2)` (see reference.md). It records
  per-level odd-padding flags in `self.last_trims`.
- `reconstruct(coeffs, trims=None)`: haar/db1 only (else `NotImplementedError`);
  `len(coeffs)` must be `level + 1` and `trims` of length `level` (else
  `ValueError`). By default it uses `last_trims` of the latest `decompose` (zeros
  if none), so it is stateful: pass `trims` explicitly (for example a saved
  `last_trims`) for other coefficients, and use one instance per length.

`UndecimatedWaveletTransform(wavelet: str = "db4", level: int = 3)`

- Same validation and buffers. `forward(x)` returns the list described above, all
  `[B, C, L]`. Circular, left-only padding of `(filter_len - 1) * 2**i`; this is
  causal-style with wraparound, not the symmetric a-trous filter. If the pad
  exceeds `L` torch raises `RuntimeError` (pinned for db4 level 3 at `L=16`),
  so require `L >= (filter_len - 1) * 2**(level - 1)`. No inverse is provided.
