---
name: "AMD"
description: "Avg-pool multi-scale mixing, patch-sequential blocks, and an MoE of MLP predictors gated by the scale embedding. Use for long-term forecasting where temporal patterns overlap at several scales; not for short lookbacks or probabilistic output."
---

# AMD

## Idea

- Multi-scale decomposable mixing average-pools the history at windows `c^k ... c` and adds each coarse scale's MLP output to the next finer scale, giving a multi-scale embedding.
- Dual dependency interaction walks the series in `patch`-wide slices: each slice aggregates the previous output slice and the input slice, with an optional `alpha`-scaled MLP across channels.
- Adaptive multi-predictor synthesis mixes `num_experts` shared `seq_len -> pred_len` MLP predictors per channel; a noisy top-k gate on the scale embedding chooses experts, and a coefficient-of-variation importance loss (`aux_loss`) balances them.

## When to use

- Long-term forecasting where several temporal scales (periods, trends) are entangled and benefit from scale-specific predictors.
- Designed around long lookbacks (official scripts use `seq_len = 512`); `seq_len` must divide by `patch` and `mix_layer_scale ** mix_layer_num`.
- Mostly channel-wise; channel mixing is optional (`alpha`, off in the preset).
- The model is tied to one `seq_len` and channel count at construction; point output only.

## Configure

- `enc_in`: number of channels (fixed at construction).
- `patch`: must divide `seq_len`.
- `mix_layer_scale`, `mix_layer_num`: `seq_len` must be divisible by `mix_layer_scale ** mix_layer_num`.

Other hyperparameters: preset defaults in `configs/models/AMD.toml`; tune generically.

## Differences

Independent rewrite checked against the paper and the pinned official code; the math follows the official modules.

- All channels are computed in one batched pass instead of the official Python loop (same shared gate and experts).
- The importance loss reproduces the official `cv_squared` and is exposed as `aux_loss` in training only.
- The official hard-coded `ff_dim`, `num_experts`, `top_k`, and BatchNorm switch are exposed as parameters; `target_slice` output selection is dropped.
- Shape validation added for `patch`, the mixing scales, and `top_k <= num_experts`.

Full detail in `reference.md`.
