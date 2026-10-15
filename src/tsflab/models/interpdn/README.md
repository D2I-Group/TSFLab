---
name: "InterPDN"
description: "Channel-independent MLP that predicts a discrete distribution per step over two interleaved support sets and returns the confidence-weighted expectation. Use for point forecasting of trend-plus-seasonal series per channel; not for calibrated probabilistic output (only the point forecast is returned)."
---

# InterPDN

## Idea

- Splits each channel into an exponential-moving-average trend and a seasonal remainder (`_ema`).
- Two `DistributionBranch`es decode seasonal and trend inputs and emit softmax logits over `support_size` bins for every horizon step; the bins are normal quantiles on two interleaved support sets (`support_first`, `support_second`).
- Each branch returns the expectation over its support and the branches are mixed by their maximum-probability confidence.
- Only the point forecast is returned (probabilities kept in `last_probabilities`).

## When to use

- Per-channel forecasting where a smooth trend plus a seasonal remainder describe the series; channels share weights and never interact.
- The dual-branch interleaved supports are meant to suppress anomalous per-step predictions.
- Forecasts (after RevIN) lie within `±support_bound` in normalized units, so very large departures from the input window's scale cannot be expressed.
- Not for tasks that need quantiles or samples: the public output is a point forecast.

## Configure

- `enc_in`: number of data channels (RevIN affine parameters and the input-shape check use it).

Other hyperparameters: preset defaults in `configs/models/InterPDN.toml`; tune generically.

## Differences

- Independent rewrite of the inference-defining Eqs. (1)-(7), checked against `models/interPDN.py` at the pinned revision; no source copied.
- Point output only; the two probability tensors are exposed as `last_probabilities`.
- The paper's patch-specific seasonal encoder is replaced by a compact residual convolutional encoder.
- The coarse-scale auxiliary branches and the four consistency losses (training-only) are not implemented.
