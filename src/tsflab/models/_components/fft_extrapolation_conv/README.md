---
name: "fft_extrapolation_conv"
description: "Zero-padded rfft convolution with per-bin complex weight and bias mapping a [B, C, L] history to a [B, C, H] horizon in O(L + H) parameters. Use for a cheap linear history-to-horizon map; not for cross-frequency interactions, variable-length histories, or dense per-step maps."
---

# fft_extrapolation_conv

## What it does

`FFTExtrapolationConv(input_length, output_length, num_sets)` forecasts a horizon
from a history with a single long circular convolution carried out in the
frequency domain. With `L = input_length`, `H = output_length`,
`g = max(floor((L + H - 1) / 100), 1)` and `T = L + H - 1 + 2g`:

`y = irfft(theta * rfft(pad(x)) + beta, n=T)[..., -(H + g):-g]`

where `pad` adds `g` zeros on the left and `H - 1 + g` zeros on the right, and
`theta` and `beta` are complex vectors with `T // 2 + 1` entries, so each
frequency bin is processed independently (a circular kernel as long as the whole
padded history plus horizon). It replaces a dense `L x H` map by
`O(L + H)` parameters and an `O((L + H) log (L + H))` computation. Note `beta`
is added in the frequency domain after the product, so it is a (circular) time
offset signal, not a per-bin scale.

## When to use

Use for a parameter-efficient linear history-to-horizon map where frequency
independence is an acceptable inductive bias. Do not use when the horizon must
depend on cross-frequency interactions, for variable-length histories (the bin
count is fixed), or when a dense per-timestep map is required.

## Interface

`FFTExtrapolationConv(input_length, output_length, num_sets=1)`

- All three sizes must be >= 1, else `ValueError`. Plain attributes
  `input_length`, `output_length`, `num_sets`, `guard`, `padded_length`,
  `num_bins`.
- Parameters `real_weight`, `imag_weight`, `real_bias`, `imag_bias`, each
  `[num_sets, num_bins]` real tensors combined into complex values on the fly
  (state-dict friendly). `real_weight[:, 0]` starts at 1 and everything else at
  0, so the initial layer outputs the same constant for every horizon step:
  `sum(x) / padded_length` (the mean of the zero-padded history, not of `x`).
- `forward(x, mixing=None)`: `x` must be `[batch, channels, input_length]`
  (else `ValueError`). With `mixing=None` the layer needs `num_sets == 1` (else
  `ValueError`) and all channels share the weight. With `mixing` of shape
  `[num_sets, channels]` channel `c` uses `sum_s mixing[s, c] * weight[s]` (and
  the same for the bias, computed by `weight_set_router.mix_weight_sets`, which
  has identical semantics to the former inline einsum); any other shape raises `ValueError`. `mixing` is cast
  to complex, and may carry gradients. Returns `[batch, channels, output_length]`
  with the real dtype and device of `x` (float32 or float64; `rfft` support for
  other dtypes is backend dependent).
- Fixed lengths: the bin count depends on `input_length`, so another history
  length is rejected. Stateless besides parameters (no buffers, no randomness).
