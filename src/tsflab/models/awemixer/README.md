---
name: "AWEMixer"
description: "Router weights undecimated wavelet subbands by spectral descriptors; temporal anchors absorb them via gated attention. Use for long-term forecasting of series mixing several periodicities with local bursts; not for very short lookbacks or cross-channel tasks."
---

# AWEMixer

## Idea

- `wavelet` (`UndecimatedWaveletTransform`) splits each channel into `wavelet_level + 1` same-length subbands, each embedded linearly.
- `FrequencyRouter` softmax-weights subbands from four z-scored descriptors: FFT band energy, wavelet energy, peak local burst energy, and spectral entropy.
- `MultiScaleTemporalEmbedding` makes one anchor per scale from convolutions of kernel `2s+1` with global pooling.
- `CoherentGatedFusion` lets anchors cross-attend the weighted frequency features and injects them through a sigmoid gate; `CrossScaleMixer` mixes scales; a linear head forecasts inside `revin`.

## When to use

- Long-term forecasting where energy is spread over several frequency bands and localized bursts matter, so an input-adaptive weighting of wavelet subbands helps.
- Channel-independent with shared weights; no cross-channel modelling.
- Needs a lookback long enough for the wavelet padding at the chosen level; point output only.

## Configure

- `enc_in`: number of channels.
- `wavelet_level`: `seq_len` must be at least `(filter_len - 1) * 2 ** (level - 1)` (db4 at level 3: 28 steps).

Other hyperparameters: preset defaults in `configs/models/AWEMixer.toml`; tune generically.

## Differences

Clean-room implementation: the Frequency Router descriptors, the Coherent Gated Fusion block and the data flow were re-derived from the pinned official file's module boundaries and comments; no lines copied.

- Wavelet filters are the cataloged orthonormal Daubechies taps; the official code divides PyWavelets' filters by an extra `sqrt(2)` and reverses them. The fixed scale is absorbed by the learned `Linear` subband embeddings.
- FFT band edges come from one `torch.linspace` call instead of the official per-band loop (numerically equivalent).
- The official `return_aux` diagnostic mode is not exposed; `last_router_weights` and `last_gates` are always recorded as attributes.
