---
name: "AROpt"
description: "iTransformer patch forecaster rolled out autoregressively; discounted patch losses penalize non-monotone error. Use for long horizons on correlated multivariate data where a short-patch rollout beats one-shot output; not for univariate or probabilistic tasks."
---

# AROpt

## Idea

- The base forecaster is an iTransformer: per-variate instance normalization, inverted variate tokens, full attention across variates, and a linear projector to one patch of `T = pred_len / rollout_steps` steps.
- Rollout (Algorithm 1): the first patch comes from the history; each later patch from the last `seq_len` values of history plus earlier predictions (fed back detached); `forward` concatenates the patches.
- Objective (Eqs. 4-5): `l = e_1 + sum_k gamma^k ((1 - beta) e_{k+1} + beta |e_{k+1} - sg(e_k)|)`, softly enforcing monotone rollout error growth.
- `training_objective` applies it per patch with the configured criterion; validation and test score the concatenated rollout.

## When to use

- Long horizons where predicting short patches autoregressively is easier than emitting the full horizon at once.
- Variate-token attention models cross-channel dependence; suited to correlated multivariate sets.
- Point output only; the default iTransformer (`d_model = 512`) is not lightweight.

## Configure

- `enc_in`: number of variates.
- `rollout_steps`: must divide `pred_len`; each call forecasts `pred_len / rollout_steps` steps.

Other hyperparameters: preset defaults in `configs/models/AROpt.toml`; tune generically.

## Differences

Independent rewrite of Sec. 3 (Algorithm 1, Eqs. 3-5) after reading the pinned official code (MIT).

- Only the paper's "AR = n" setting: the rollout covers exactly the evaluated horizon; the headline "AR = 1" rows (training beyond the horizon) are not reproduced.
- Each call reads the last `seq_len` values of history plus predictions (Algorithm 1); the official code feeds only the previous patch.
- Per-patch error uses the configured criterion; calendar marks are not added as variate tokens.
- Only the iTransformer backbone (the paper also trains iInformer, iFlowformer, iFlashformer).

Full detail in `reference.md`.
