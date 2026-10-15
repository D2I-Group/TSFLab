---
name: "STLoRA"
description: "LSTM backbone whose forecast is refined by node-specific predictors built from per-node low-rank (LoRA) layers and blended by a learned gate. Use for spatio-temporal node forecasting where nodes behave heterogeneously, on equal input and output lengths; not for seq_len != pred_len or probabilistic output."
---

# STLoRA

## Idea

- `NodeAdaptiveLowRankLayer` (Eq. 2): a frozen shared projection `W x + b` plus `(alpha / r) B A_i x`, with `B` shared and `A_i` one rank-`r` factor per node (Kaiming-initialised `A`, zero `B`, dropout on the low-rank path).
- `NodeSpecificPredictor` (Eq. 3): a 1x1 `Conv2D` lift, `L` residual layers `H_l = H_{l-1} + NALL(sigma(H_{l-1}))` with sigma = RMSNorm, LeakyReLU and dropout, and a NALL head to one feature per step and node.
- `Model.adapt` / `Model.fuse` (Eq. 4): the first predictor reads `relu(Y_base)`, later ones `relu([X, Z_{k-1}])`, and `R = sigmoid(F([X, mean Z]))` blends `R * Y_base + (1 - R) * mean Z`.
- `LSTMBackbone` is the base forecaster (pointwise input projection, multi-layer LSTM per node, ReLU MLP to all horizon steps); backbone and adapters train jointly.

## When to use

- Node data (traffic and similar sensor networks) where a shared backbone underfits node-specific behaviour; per-node low-rank factors add heterogeneity cheaply.
- The 12-to-12 style protocol: history and adapter outputs are concatenated step by step, so `seq_len` must equal `pred_len`.
- No graph is used. Point forecasts only.

## Configure

- `enc_in` follows the node count; it must equal it exactly (one `A_i` per node).
- `seq_len` must equal `pred_len` (enforced).
- `cov_dim` follows the loader's covariate features (input is value plus `cov_dim` covariates).
- Other hyperparameters: preset defaults in `configs/models/STLoRA.toml` (rank 16, four NALL layers, one block, Sec. 4.2); tune generically.

## Differences

Independent rewrite of Sec. 3.1-3.3 (arXiv v2) after reading `RWLinno/ST-LoRA@c89c21d` (MIT); nothing copied. Recorded in `card.toml` issues:

- Follows the paper, not the code: node-specific `A_i`, residual RMSNorm layers, chained blocks with the learned gate (official: shared `A`, no residuals, a different fusion).
- Official node/step reshape bug is not reproduced.
- The backbone trains jointly from scratch (official default), not pre-trained as the paper describes; only the LSTM backbone is provided.
- The `lambda ||alpha||^2` gate penalty of Eq. (5) is not added (use weight decay); the official per-module optimisers and StepLR are left to the trainer.

Full detail in `reference.md`.
