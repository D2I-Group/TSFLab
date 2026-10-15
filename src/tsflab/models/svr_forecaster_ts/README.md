---
name: "SVRForecasterTS"
description: "Differentiable epsilon-SVR baseline: RBF features of each channel's lookback against learned support centres, linear readout shared across channels. Use as a lightweight nonlinear per-channel baseline on small or short datasets; not for cross-channel structure, trending data, or probabilistic output."
---

# SVRForecasterTS

## Idea

- Treats each channel's lookback as a query and computes Gaussian RBF features against `num_support` learned support centres (`kernel_gamma`).
- A linear map plus bias over the kernel features gives the horizon; all channels share the same centres and coefficients.
- `forward` stores an L2 penalty on the coefficients in `aux_loss`; `epsilon_insensitive_loss` provides the SVR objective as a helper, while the registered training loss stays MSE.

## When to use

- A classical kernel-regression reference point on small datasets or short training windows, where a few hundred parameters suffice.
- Channels that behave alike: one shared kernel expansion serves every channel, with no cross-channel interaction.
- Not for data with strong trends or level shifts (no normalization; RBF features saturate far from the centres), or for probabilistic output.

## Configure

- `enc_in`: number of channels; input must be `[B, seq_len, enc_in]`.
- `kernel_gamma`: the squared distance sums over all `seq_len` steps, so a suitable value shrinks as `seq_len` (and the data scale) grows.

Other hyperparameters: preset defaults in `configs/models/SVRForecasterTS.toml`; tune generically.

## Differences

- A clean-room differentiable adaptation, not a convex SVR solver: support centres and coefficients are optimized directly by gradient descent, with no dual constrained optimization.
- The standard trainer uses the epsilon-insensitive loss only when explicitly configured to call the helper.
- No residual linear head; no third-party implementation was inspected or copied.
