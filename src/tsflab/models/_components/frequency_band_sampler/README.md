---
name: "frequency_band_sampler"
description: "Map a layer index to a contiguous half-open FFT bin range (layer 0 highest frequencies, last layer near DC) and slice a spectrum to it. Use for giving each encoder depth a fixed, deterministic frequency slice (Dualformer); not for learned band boundaries (freq_band_moe) or a frequency axis other than dim 1."
---

# frequency_band_sampler

## What it does

`HierarchicalFrequencySampler(num_layers, alpha)` allocates a spectral band to
each network depth: shallow layers get high frequencies, deep layers get low
frequencies (layer `num_layers - 1` includes bin 0). For `n = num_bins`, `L =
num_layers`, layer `i`:

- tiling regime (`L == 1` or `alpha <= 1/L`): shared integer edges
  `e_k = (n * k) // L`; layer `i` is `[e_{L-1-i}, e_{L-i})`. The bands partition
  `[0, n)` exactly (no gaps, no overlap, layer 0 ends at `n`, layer `L-1` starts at 0)
  whenever `n >= L`.
- sliding regime (`alpha > 1/L`): `start = int(n * (1 - alpha) * (1 - i/(L-1)))`,
  `end = start + ceil(alpha * n - 1e-9)` (rounded up, with a 1e-9 float guard, so
  layer 0 reaches the last bin and layer `L-1` starts at bin 0); adjacent layers
  overlap. `alpha=1.0` gives every layer the full range `[0, n)`.
- `end` is clipped to `n` and to at least `start + 1`; the range is `[start, end)`.
  With `n < L` the tiling cannot avoid empty bands, so each band keeps one bin and
  bands overlap.

## When to use

Use to give each depth a fixed, deterministic frequency slice of a
`[batch, freq, ...]` spectrum (high to low). Do not use when bands must be
learned (see `freq_band_moe` for learned boundaries),
or when the frequency axis is not dim 1.

## Interface

`HierarchicalFrequencySampler(num_layers: int, alpha: float = 1.0)`

- `num_layers` (int >= 1), `alpha` (float in `(0, 1]`); otherwise `ValueError`.
  Default `alpha=1.0` selects the sliding regime for `num_layers > 1`, i.e.
  full-width windows (every layer gets all bins).
- `band(num_bins, layer_idx) -> (start, end)`: `num_bins >= 1` and
  `0 <= layer_idx < num_layers`, else `ValueError`. Pure Python ints.
- `sample(spectrum, layer_idx) -> Tensor`: any real or complex `spectrum.ndim >= 2` (typically `rfft` output) with the
  frequency axis at dim 1; returns `spectrum[:, start:end]` (a view, dtype and
  device preserved). `ValueError` for `ndim < 2`.
- No parameters, buffers, or state; attributes `num_layers`, `alpha`.
- Examples for 17 bins: `L=3, alpha=0.3` gives `(11,17), (5,11), (0,5)`;
  `L=2, alpha=0.3` gives `(8,17), (0,8)`.
