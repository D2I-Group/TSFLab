---
name: "spectral_descriptor"
description: "Parameter-free window statistics: normalized spectral entropy and low/mid/high band-energy ratios of the channel-averaged power spectrum. Use for conditioning gates or routers on how periodic or noisy a window is (CoRe); not for per-channel spectra, very short windows, or physical band edges."
---

# spectral_descriptor

## What it does

`SpectralDescriptor(eps=1e-8)` summarizes how a window's spectral energy is
distributed. Per sample, with each channel's temporal mean removed:

- `P = mean_c |rfft(x)|^2` over `K = length // 2 + 1` bins;
- `p = (P + eps) / (sum P + K * eps)`;
- entropy `H = -sum p log(p + eps) / log(max(K, 2))`;
- band ratios: sums of `p` over bins `[0, K//3]`, `[K//3 + 1, 2K//3]`,
  `[2K//3 + 1, K - 1]` (all inclusive).

It returns `[H, low, mid, high]`; the three ratios sum to one and `H` lies in
`[0, 1]` up to `eps`-level rounding. For very short windows the upper bands can be empty and
contribute zero (`K = 1`: mid and high; `K = 2` or `3`: high). Low entropy
means energy is concentrated in few frequencies; the band ratios say where.

## When to use

Use as a cheap, parameter-free conditioning signal for gates, routers, or
schedulers that should react to how periodic or noise-like a window is. Do not
use when per-channel spectra matter (channels are averaged), for very short
windows (few bins make the thirds coarse), or when the band split must follow
known physical frequencies.

## Interface

`SpectralDescriptor(eps: float = 1e-8)`

- `eps` (float > 0): smoothing added to every bin; raises `ValueError` otherwise.
- `forward(x)`: floating `[batch, length, channels]`; raises `ValueError` when
  `x.ndim != 3`. Returns `[batch, 4]` on the device and real dtype of `x`.
- No parameters, buffers, or state; fully differentiable (no argmax).
- Channels are averaged before normalization, so the output describes the
  window, not each channel.
