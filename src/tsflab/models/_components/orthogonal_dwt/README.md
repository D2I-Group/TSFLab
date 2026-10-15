---
name: "orthogonal_dwt"
description: "Multi-level orthogonal DWT and inverse along the last axis in the PyWavelets convention, zero or symmetric boundary, haar/db4/sym3/coif3 filters. Use for wavelet multi-scale models whose official code calls pywt or pytorch_wavelets; not for learnable, undecimated, circular-boundary, or biorthogonal transforms."
---

# orthogonal_dwt

## What it does

`OrthogonalDWT` is the critically sampled orthogonal discrete wavelet transform
in the convention of PyWavelets and `pytorch_wavelets`. One analysis level maps
`x [..., N]` to

`a[n] = sum_j dec_lo[j] x[2n + 1 - j]`, `d[n] = sum_j dec_hi[j] x[2n + 1 - j]`,

with `x` extended outside `[0, N)` by the selected boundary mode, so each band has
`floor((N + L - 1) / 2)` coefficients for an `L`-tap filter. One synthesis level is
the transposed (adjoint) operation with the reconstruction filters `rec = reversed(dec)`,
keeping `2n - L + 2` samples. `decompose`/`reconstruct` iterate the levels on the
approximation. Its coefficients agree with `pywt.dwt`/`pywt.wavedec` and
`pywt.idwt`/`pywt.waverec` for the same wavelet and mode (`zero` or `symmetric`).

## When to use

Use when a model separates a series (or embedding) into coarse and fine scales
with an orthogonal wavelet and must match PyWavelets / `pytorch_wavelets`
coefficients in `zero` or `symmetric` mode, or when an invertible multi-level DWT
with a known boundary rule is wanted. Do not use for learnable filters,
undecimated (a-trous) or same-length bands (see `wavelet`), circular boundaries,
or biorthogonal wavelets.

## Interface

- `OrthogonalDWT(wavelet, levels=1, mode="zero", persistent=True)`: `wavelet` is a
  key of `DEC_LO` (`haar`, `db4`, `sym3`, `coif3`; else `ValueError`); `levels >= 1`;
  `mode` in `BOUNDARY_MODES = ("zero", "symmetric")`. Attributes `wavelet`,
  `levels`, `mode`, `filter_length` (`L`).
- Buffers: `analysis` `[2, 1, L]` (reversed `dec_lo`, `dec_hi`; `conv1d`
  correlates) and `synthesis` `[2, 1, L]` (`rec_lo`, `rec_hi`), registered in that
  order, float32 at construction and cast to the input dtype at call time; in
  `state_dict()` only when `persistent=True`. No parameters.
- `analyze(x) -> (a, d)`: `x [..., N]`, any leading shape; each output
  `[..., floor((N + L - 1) / 2)]`. `mode="symmetric"` raises `ValueError` when
  `N < L - 1`.
- `synthesize(a, d) -> [..., 2n - L + 2]` for `a, d [..., n]`; the exact inverse of
  `analyze` for even `N` (for odd `N` crop one trailing sample after dropping the
  approximation overhang, as `reconstruct` does).
- `decompose(x) -> [d_1, ..., d_J, a_J]` (finest detail first, the reverse of
  `pywt.wavedec`).
- `reconstruct([d_1, ..., d_J, a_J]) -> [..., 2 len(d_1) - L + 2]`: drops the last
  sample of an approximation that is one longer than the next detail, then
  synthesizes; the result is the input length or one longer (odd input), so crop
  it. Wrong list length raises `ValueError`.
- `coefficient_length(N)` (method) and module functions
  `coefficient_length(N, L)`, `coefficient_lengths(N, L, J)` (lengths of
  `[d_1, ..., d_J, a_J]`), `filter_bank(name) -> (dec_lo, dec_hi, rec_lo, rec_hi)`
  in float64 with `dec_hi[k] = (-1)^(k+1) dec_lo[L-1-k]`, and the `DEC_LO` table.
