---
name: "harmonic_energy_gate"
description: "Parameter-free per-channel share of spectral energy on the dominant low-frequency fundamental and its harmonics, a periodicity gate in [0, 1]. Use for weighting frequency against time branches per channel on periodic data; not for very short windows, aperiodic signals with an ambiguous fundamental, or a learned gate."
---

# harmonic_energy_gate

## What it does

`HarmonicEnergyGate(num_harmonics=3, low_freq_guard=3)` measures how periodic
each channel is. With `A = |rfft(x - mean_t x)|` (`nb = length // 2 + 1` bins):

- candidates are bins in `[low_freq_guard, max(low_freq_guard, nb // num_harmonics))`;
  `f0 = argmax` of `A` over that range, per sample and channel;
- `E_h = sum_{k=1..num_harmonics} A[min(k * f0, nb - 1)]^2`;
- output `E_h / (sum_f A[f]^2 + 1e-5)`.

A channel whose energy sits on one fundamental and its harmonics scores near
1; diffuse spectra score near 0.

## When to use

Use to weight a frequency-domain branch against a time-domain branch per
channel from a `[B, L, C]` tensor. Do not use for very short sequences relative
to `low_freq_guard * num_harmonics`, for signals with aperiodic structure where
the low-frequency fundamental is ambiguous, or when a learned gate is needed
(this one has no parameters).

## Interface

`HarmonicEnergyGate(num_harmonics: int = 3, low_freq_guard: int = 3)`

- `num_harmonics` (int >= 1): fundamental plus the next `num_harmonics - 1`
  multiples. `low_freq_guard` (int >= 1): bins below this index (including DC)
  are never the fundamental; `dualformer` passes its `dc_bins` here.
- `forward(x)`: floating `[batch, length, channels]`. Raises `ValueError` for
  `ndim != 3` and when `length // 2 + 1 <= low_freq_guard`. Returns
  `[batch, 1, channels]` on the same device and real dtype as `x`.
- No parameters, buffers, or state. The argmax selection is not differentiable;
  gradient flows only through the amplitudes at the selected bins and the total.
- Quirk: when `nb // num_harmonics <= low_freq_guard` the candidate range is
  empty, every candidate is zero, `f0` is the tied argmax (index 0 in practice), and the output is about 0 (DC removed), so
  short inputs or many harmonics silently yield a zero gate.
