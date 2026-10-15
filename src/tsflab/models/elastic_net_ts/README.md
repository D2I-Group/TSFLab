---
name: "ElasticNetTS"
description: "Shared linear lag regression from lookback to all horizons, gradient-trained with an L1/L2 elastic-net weight penalty. Use as a cheap regularized linear baseline, especially on short training windows; not for nonlinear dynamics or cross-channel effects."
---

# ElasticNetTS

## Idea

- A single `nn.Linear(seq_len, pred_len)` maps each channel's lags to all horizons, with coefficients shared across channels.
- The forward pass sets `aux_loss = penalty * (l1_ratio * ||w||_1 + 0.5 (1 - l1_ratio) ||w||_2^2)`, the elastic-net penalty (Lasso sparsity plus Ridge stability, with grouping of correlated lags).
- It is gradient-trained: no least-squares solution path or variable-selection procedure.

## When to use

- A sanity baseline: if a deep model cannot beat it, the extra capacity is not paying off.
- The penalty keeps the many correlated lag coefficients stable and sparse, which helps when training windows are few.
- Linear and channel-independent: misses nonlinear dynamics and cross-channel structure; no instance normalization, so level shifts between splits hurt it.

## Configure

- `enc_in`: number of channels (shape check only; weights are shared).

Other hyperparameters: preset defaults in `configs/models/ElasticNetTS.toml`; tune generically.

## Differences

Clean-room, gradient-optimized direct multi-horizon adaptation of Zou and Hastie (2005); no third-party implementation inspected or copied.

- Coefficients are shared across channels.
- The paper's least-squares solution path (LARS-EN) and variable-selection experiments are not reproduced.

Citation: Zou, H., Hastie, T. "Regularization and Variable Selection via the Elastic Net." JRSS Series B 67(2), 301-320 (2005). doi:10.1111/j.1467-9868.2005.00503.x.
