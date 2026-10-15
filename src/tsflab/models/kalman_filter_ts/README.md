---
name: "KalmanFilterTS"
description: "Differentiable alpha-beta level/velocity filter with learned per-channel gains and linear extrapolation. Use as a tiny baseline for smooth, locally trending series or very short training data; not for seasonal or cross-channel structure."
---

# KalmanFilterTS

## Idea

- Runs a recursive level/velocity update over the window with innovation gains `alpha` and `beta` (sigmoid of learnable logits, one pair per channel).
- The forecast extrapolates `level + h * velocity` for horizon step `h`.
- A fixed-gain specialization with no covariance recursion, unit time steps, and only `2 * enc_in` parameters.

## When to use

- Smooth series dominated by a local level and trend; a sanity baseline with almost no capacity.
- Very short training windows or tight compute budgets.
- Not for seasonal data: the forecast is a straight line from the last level and velocity.
- Channels are filtered independently; no cross-channel information is used.

## Configure

- `enc_in`: number of data channels; one gain pair is learned per channel.

Other hyperparameters: preset defaults in `configs/models/KalmanFilterTS.toml`; tune generically.

## Differences

- Clean-room fixed-gain alpha-beta specialization, not the cited paper's full covariance-based Kalman filter; no third-party implementation was inspected or copied.
- Gains are learned directly (bounded by a sigmoid); unit time steps, a constant-velocity state, and no exogenous control are assumed.
- Citation: R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems", Journal of Basic Engineering 82(1):35-45, 1960, doi:10.1115/1.3662552.
