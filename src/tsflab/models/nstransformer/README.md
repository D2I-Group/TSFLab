---
name: "NSTransformer"
description: "Encoder-decoder Transformer with series stationarization and de-stationary attention (learned tau and delta) that restores non-stationary information. Use for multivariate point forecasting on series whose mean and scale drift across windows; not for exogenous or calendar inputs, or probabilistic output."
---

# NSTransformer

## Idea

- The input is standardized with detached per-window mean and standard deviation, and the statistics are restored on the output (series stationarization).
- Two `Projector` MLPs (`tau_learner`, `delta_learner`) learn a scale `tau` and a shift `delta` from the raw series and its statistics.
- `DeStationaryAttention` computes `softmax((tau QK^T + delta)/sqrt(d))`, so attention on the stationarized input recovers the non-stationary information (against over-stationarization).
- A learned future-query decoder (`future_queries`) attends to the encoded history; `projection` maps tokens back to channels.

## When to use

- Non-stationary series whose window mean and scale drift over time, where plain normalization would make attention maps indistinguishable across series.
- Multivariate data where a channel-mixing token (all channels embedded per time step) is acceptable.
- Not when timestamps or exogenous covariates carry signal (marks are ignored), when quantiles are needed (point output only), or for very long lookbacks (full attention over `seq_len` tokens).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count (sizes the value embedding, projectors and output projection).

Other hyperparameters: preset defaults in `configs/models/NSTransformer.toml`; tune generically.

## Differences

- Clean-room implementation of the paper's structure: detach-and-normalize stationarization, statistic projectors, de-stationary attention, statistic restoration. The MIT source is reference-only; no code or weights are reused.
- The decoder uses learned future queries instead of a label-window decoder input; `x_dec` and time marks are accepted and ignored.
- Positions are a learned parameter of length `seq_len`; `tau` is predicted in log space and clamped to `[-5, 5]` before `exp`.
