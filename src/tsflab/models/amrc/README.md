---
name: "AMRC"
description: "Training objective adding adaptive prefix-masking loss and embedding-similarity penalty to a compact linear forecaster. Use for long lookbacks with redundant or noisy history; not for cross-channel modelling (the carrier is a per-channel linear model)."
---

# AMRC

## Idea

- AMRC is an optimization method; the forecaster is a compact carrier (linear encoder with `GELU` and `LayerNorm`, then a linear predictor) applied per channel, wrapped by `revin`.
- Adaptive masking loss masks the oldest prefix of the lookback at several lengths, keeps the best mask per sample, and pulls the unmasked embedding toward it, weighted by the relative improvement.
- Embedding similarity penalty matches pairwise embedding distances to pairwise target distances within a batch.
- `training_objective` sums MSE with both terms (`lambda_aml`, `lambda_esp`); plain `forward` is point forecasting only.

## When to use

- Long lookback windows where truncating old history improves accuracy, i.e. the model would otherwise fit redundant noise or irrelevant fluctuations.
- Channel-independent carrier: no cross-channel modelling.
- The gain comes from the training objective, which the spec declares so the trainer uses it while training; validation and test use plain `forward`.

## Configure

- `enc_in`: number of channels.
- No other data-dependent parameter; mask candidates are prefix lengths inside `seq_len`.

Other hyperparameters: preset defaults in `configs/models/AMRC.toml`; tune generically.

## Differences

Clean-room implementation of paper Eqs. (6)-(14) (sampled prefix selection, improvement-weighted embedding alignment, pairwise embedding/output geometry matching, the combined objective); the official repository (no license file) was neither inspected nor copied. The paper is backbone-agnostic, so this entry supplies a compact channel-independent carrier. Default mask candidates are evenly spaced unless the caller supplies stochastic lengths. The spec declares `training_objective`, so the trainer applies AML and ESP during training; validation and test metrics use `forward`.
