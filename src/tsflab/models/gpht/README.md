---
name: "GPHT"
description: "Auto-regressive token forecaster: max-pooled causal patch-Transformer stages chained by residuals, outputs summed, rolled out token by token. Use for channel-independent forecasting from long lookbacks to arbitrary horizons; not for cross-channel or covariate tasks."
---

# GPHT

## Idea

- Auto-regressive over tokens of `token_len` steps: a causally masked patch Transformer maps every token to the next token (`HierarchicalStage`, Eq. 3).
- Each stage max-pools its input (`pooling_rates`, default 8/4/2/1) so early stages see coarse patterns; stage inputs are iterative residuals `x - PadFirstToken(out)` and stage outputs are summed (Eqs. 4-5).
- `revin` wraps the stack; `forward` rolls the window one token at a time for horizons longer than `token_len`, so one trained model serves any horizon.
- Training is the paper's teacher-forced next-token MSE over every position (`training_objective`; the configured criterion is not used).

## When to use

- Designed for long input windows (official default `seq_len = 336`) split into tokens, with coarse-to-fine stages capturing patterns at several temporal scales.
- One model forecasts any horizon by rollout; errors can compound over long rollouts.
- Channel-independent: no cross-channel modelling; no calendar or covariate inputs.
- This entry is the architecture only; the mixed-dataset pretraining that motivates the paper is not included.

## Configure

- `enc_in`: the dataset's channel count (training needs all channels as targets; `MS` slicing is not supported).
- `token_len`: `seq_len` must be a multiple of it, training needs `pred_len >= token_len`, and every pooling rate must divide it.

Other hyperparameters: preset defaults in `configs/models/GPHT.toml`; tune generically.

## Differences

- Local rewrite from Section 3 (Eqs. 3-6) after inspecting the official code (MIT) at the pinned revision; per-stage pooling and patch embedding, causal encoder, token head, per-window normalization, one-token residual shift and the training target follow the code.
- `depth` is `len(pooling_rates)`; the objective accepts any `pred_len >= token_len` (officially `pred_len == token_len`) and `forward` rolls out for any `pred_len`.
- Not implemented: the pretraining corpus and checkpoint and the fine-tuning mode that freezes all but `forecast_head`.
- Preset uses the official pretraining defaults; no training was run, and reported benchmark numbers are not reproduction claims.

Full detail: `reference.md`.
