---
name: "TimeBridge"
description: "Patch Transformer that detrends queries/keys for attention within each variate but keeps non-stationarity when attending across variates (cointegration). Use for non-stationary multivariate series with long-term co-movement (e.g. financial indices); not for aperiodic data or calendar-driven targets."
---

# TimeBridge

## Idea

- Each variate is cut into patches of one `period`; `IntegratedAttention` attends over a variate's patch tokens with queries and keys from the tokens minus a moving-average trend (`stable_len`), so short-term non-stationarity leaves the scores while values keep it.
- `PatchDownsample` pools queries to `long_count` tokens and cross-attends to the full patch sequence to aggregate long context.
- `CointegratedAttention` attends across variates at each downsampled position on the raw (non-stationary) tokens to model long-term cointegration.
- `revin` normalizes the input; a flatten linear layer maps tokens to the horizon.

## When to use

- Multivariate series with short-term fluctuations and long-term trends, where removing non-stationarity everywhere would hide cross-variate cointegration; the paper also reports strong results on CSI 500 and S&P 500.
- Data with a known period that sets the patch length.
- Moderate channel counts: cross-variate attention is quadratic in `enc_in`.
- Not when calendar marks or covariates matter (not used), or for probabilistic output.

## Configure

- `enc_in`: number of channels.
- `period`: patch length in steps; the dataset's dominant period (24 for hourly daily cycles).
- `num_p`: number of period patches; defaults to `ceil(seq_len / period)`. A smaller value uses only the last `num_p * period` steps; a larger one replicate-pads the lookback on the left.
- `stable_len`: moving-average window for detrending queries/keys; also sets `long_count = num_p // stable_len` downsampled tokens.

Other hyperparameters: preset defaults in `configs/models/TimeBridge.toml`; tune generically.

## Differences

- Paper-driven local implementation of Equations (3)-(10); the external repository is reference-only and no source was copied or adapted.
- Calendar marks are not part of this local contract.
