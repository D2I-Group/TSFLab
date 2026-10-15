---
name: "haar_dwt1d"
description: "Stateless one-level orthonormal Haar DWT along the last axis returning (approx, detail) of length ceil(T/2), with an exact inverse that can truncate odd-length padding. Use for an exact one-level Haar split of [..., T] tensors; not for deeper pyramids or other wavelets (wavelet) or time-aligned sub-series."
---

# haar_dwt1d

## What it does

`HaarDWT1D` rotates adjacent sample pairs `(a, b)` into
`approx = (a + b) / sqrt(2)` and `detail = (a - b) / sqrt(2)`.
`HaarIDWT1D` inverts it: `even = (approx + detail) / sqrt(2)`,
`odd = (approx - detail) / sqrt(2)`, interleaved. The transform is orthonormal
and lossless; an odd-length input is first extended by replicating its last
sample, and the inverse can truncate back with `length`.

## When to use

Use for a stateless, exact one-level Haar split of `[..., T]` tensors where the
caller tracks the original length. Do not use for deeper pyramids or
non-Haar filters (use `wavelet`), or when sub-series must stay aligned
with the time axis (the outputs are decimated by 2).

## Interface

`HaarDWT1D()` and `HaarIDWT1D()`, neither takes constructor arguments.

- `HaarDWT1D.forward(x) -> (approx, detail)`: `x` floating, any leading shape,
  last axis `T >= 2` (else `ValueError`). Both outputs have last length
  `ceil(T / 2)`; leading axes, dtype, and device are preserved.
- `HaarIDWT1D.forward(approx, detail, length=None) -> Tensor`: shapes must be
  identical (else `ValueError`, no broadcasting); output last length
  `2 * approx.shape[-1]`, then sliced to `[:length]` when given (pass the
  original `T` for odd inputs; a `length` larger than the output is not
  rejected and returns the full reconstruction).
- No parameters, buffers, or state-dict keys; no stored padding record, so the
  caller must remember the original length.
