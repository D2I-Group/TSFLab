---
name: "GOTSF"
description: "Goal-oriented forecasting: interval-conditioned predictions with membership confidences, reweighted at inference to emphasize a chosen value range without retraining. Use when accuracy matters most in an application-specific value range; not for cross-channel modelling or calibrated probabilistic output."
---

# GOTSF

## Idea

- Splits the value range `[interval_min, interval_max]` into `num_intervals` disjoint bins; each bin's midpoint and half-width condition a shared channel-independent MLP (`history_encoder` plus `interval_encoder`).
- A regression head and a membership-confidence head produce a forecast and a confidence per interval (`interval_outputs`).
- `forecast_interval` averages the forecasts of bins intersecting a target interval (Eq. 13), weighted by confidence, so the emphasized range can change without retraining; plain `forward` uses all bins.
- `goal_oriented_loss` trains one bin with an exponential soft-boundary weight (Eq. 8, `decay_rate`) on an L1 term plus a membership BCE term; the training objective averages it over all bins, while validation and test score `forward` with the standard loss.

## When to use

- Designed for downstream decisions that depend on a specific value range (the paper's wireless-communication case), where overall error is less important than error inside a region of interest.
- The region can be changed at inference via `forecast_interval`; plain benchmark runs use all bins.
- Channel-independent: no cross-channel modelling.
- Not a calibrated probabilistic forecaster: confidences are bin memberships, not quantiles.

## Configure

- `enc_in`: the dataset's channel count.
- `interval_min`, `interval_max`: edges of the binned value range in the scaled data (preset `-2`, `2` for standardized series); they should cover the training range of the targets, `interval_max > interval_min`.

Other hyperparameters: preset defaults in `configs/models/GOTSF.toml`; tune generically.

## Differences

- Independent implementation of Eqs. (8)-(14) after inspecting the pinned official `experiments/exp.py` and `experiments/exp_long_term_forecasting_discrete.py`; nothing copied.
- The training objective averages the per-bin loss over all bins each step (the official trainer samples one random bin per example), with element-wise soft-boundary weights on L1 and an unweighted BCE as officially.
- A latent interval embedding replaces the paper experiment's repeated interval-bound channels.
- The paper permits multiple host forecasters; this entry uses a compact channel-independent MLP and does not claim the paper's dataset recipes or reported accuracy.
