---
name: "SAMformer"
description: "One channel-wise attention block plus a shared linear head under RevIN, trained with sharpness-aware minimization. Use for lightweight multivariate forecasting where cross-channel attention helps; not for probabilistic output or tight training budgets (SAM doubles each step)."
---

# SAMformer

## Idea

- `ChannelWiseAttention` treats each variate as a token whose features are its `seq_len` time steps: `softmax(QK^T / sqrt(hid_dim)) V` with `Q, K` projected to `hid_dim` and `V` kept at width `seq_len`, added residually; one head, one layer.
- One shared linear layer (`channel_wise_linear`) maps the attended window to the horizon; `revin` normalizes the input and restores scale on the forecast.
- Sharpness-aware minimization is the training procedure, not architecture: `spec.training_objective` evaluates the loss at `w + rho * g / ||g||` through the `sharpness_aware` component; `forward` is the plain network.

## When to use

- Multivariate series whose channels inform each other: attention runs across channels, with time handled by one linear map per window.
- Small, cheap model (one attention layer, one linear head) when a deep Transformer would be hard to train.
- The `C x C` attention map grows quadratically with channel count; on weakly correlated channels a channel-independent linear model is the simpler baseline.
- Training costs two forward/backward passes per step (SAM); point forecasts only.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly (input shape and RevIN affine parameters are checked).
- Other hyperparameters: preset defaults in `configs/models/SAMformer.toml`; tune generically.

## Differences

Architecture matches the official `SAMFormerArchitecture` at `71f10ea` (independent rewrite). Main differences:

- SAM is a `training_objective` (loss at perturbed weights via `torch.func.functional_call`), not the official two-step optimizer wrapper; the run's own optimizer, scheduler, and clipping apply.
- `hid_dim` is configurable (official trainer hard-codes 16).
- Output is `[batch, pred_len, channels]` on the framework data pipeline, not the official flattened tensor and 100-epoch loop.

Full detail in `reference.md`.
