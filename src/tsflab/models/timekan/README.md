---
name: "TimeKAN"
description: "Lightweight frequency-decomposition forecaster: pyramid bands, each learned by a Chebyshev KAN of band-specific order plus a depthwise conv, remixed coarse to fine. Use for long-term forecasting of series with several intertwined frequency components; not for cross-channel structure or covariates."
---

# TimeKAN

## Idea

- Cascaded frequency decomposition: an average-pooling pyramid; each band is a level minus the FFT-upsampled next coarser level (`frequency_upsample`, zero-padding the spectrum).
- Multi-order KAN (`MultiOrderKAN`): a `ChebyshevKAN` whose polynomial order differs per band (`begin_order` + level offset) plus a circular depthwise convolution.
- Frequency mixing: bands are recombined from coarse to fine by adding upsampled mixed outputs.
- The finest level goes through a linear readout and a `Linear(seq_len, pred_len)` forecast; each variable is processed independently under `revin`.

## When to use

- Long-term forecasting of series whose frequency components are intertwined and carry different information density (e.g. daily plus slower cycles), so a uniform model fits them poorly.
- Very tight compute or parameter budgets (`d_model = 16` by default).
- Not when cross-channel interactions, timestamps, or covariates drive the target (channel-independent; marks ignored).

## Configure

- `enc_in`: number of channels (sizes RevIN); `c_out`, if given, must equal `enc_in`.
- `down_sampling_window`, `down_sampling_layers`: the pyramid pools `seq_len` by `down_sampling_window` per level; `seq_len` should be divisible by `down_sampling_window ** down_sampling_layers` (pooling drops remainders).

Other hyperparameters: preset defaults in `configs/models/TimeKAN.toml`; tune generically.

## Differences

- Paper-driven local implementation of Equations (3)-(12); the external repository is reference-only and no source was copied or adapted.
- Timestamp covariates are not part of the active path; `moving_avg`, `dropout`, `embed` and `freq` are accepted for config compatibility but ignored.
