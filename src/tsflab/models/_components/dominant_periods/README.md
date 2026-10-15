---
name: "dominant_periods"
description: "Top-k FFT period selection: integer periods T // f from the batch-mean amplitude spectrum, plus per-sample amplitudes at those frequencies. Use for multi-period models (TimesNet-style folding) on seasonal data; not for per-sample periods (they are batch-shared), fully on-device calls, or frequency-domain gating."
---

# dominant_periods

## What it does

`dominant_periods(x, k=2)` finds the `k` strongest non-DC frequencies of a
batch of series and converts them to integer periods. With
`S = rfft(x, dim=1)` (shape `[B, T//2+1, C]`):

- `strength[f] = mean_b mean_c |S[b, f, c]|`, with `strength[0] = 0` (DC removed).
- `idx = topk(strength, k)` (descending strength); `periods = T // idx`.
- `amplitudes[b, j] = mean_c |S[b, idx[j], c]|`.

The periods are one set shared across the whole batch; the amplitudes are
per-sample and keep the autograd graph, the periods do not.

## When to use

Use for multi-period models that fold or group a `[B, T, C]` window by a small
number of dominant cycles. Do not use when periods must differ per sample (they
are batch-shared), when the whole call must stay on-device (it returns a numpy
array), or when a frequency-domain gate is needed instead of integer periods
(see `harmonic_energy_gate`).

## Interface

`dominant_periods(x: Tensor, k: int = 2) -> tuple[np.ndarray, Tensor]`

- `x`: floating tensor `[batch, time, channels]`; `ndim != 3` raises `ValueError`.
- `k`: int in `[1, T//2]` (number of non-DC bins); otherwise `ValueError`.
- Returns `(periods, amplitudes)`: `periods` is a numpy `int64` array of shape
  `[k]` (computed on CPU via `.detach().cpu().numpy()`, so each call forces a
  device sync); `amplitudes` is a tensor `[batch, k]` on the input device with
  gradient to `x`.
- No parameters, buffers, or state. The `rfft` is computed twice internally.
- Edge case: if the batch-mean spectrum is all zero (for example a constant
  zero input after DC removal) `topk` may return index 0, which gives a numpy
  divide-by-zero warning and a period of 0; callers must guard `period >= 1`
  (MSGNet clamps with `max(1, ...)`; TimesNet does not).
