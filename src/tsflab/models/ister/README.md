---
name: "Ister"
description: "Inverted Transformer with linear-cost Dot-attention over seasonal variate tokens plus a residual-MLP trend path. Use for multivariate forecasting with many channels where full variate attention is costly; not for non-hourly data with calendar tokens (marks only rebuild for freq h)."
---

# Ister

## Idea

- `DotAttention` (Eq. 1): `Softmax(Q)` over variates, `sum_i Softmax(Q)_i ⊙ K_i` gives one global `[D]` vector that gates every value token, so cost is linear in the number of variates; it sits in a single-head `AttentionLayer` inside post-norm `EncoderLayer`s.
- Algorithm 1: non-affine instance normalization, `SeriesDecomposition`, separate `Linear(T -> D)` inverted embeddings for seasonal and trend parts, the Dot-attention encoder on seasonal tokens, and `h + TrendMLP(h)` on trend tokens.
- Summed `seasonal_head` and `trend_head` (`Linear(D -> recon_len + pred_len)`, initialized to `1 / D`); `forward` returns the last `pred_len` steps.
- `training_objective` adds the reconstruction of the last `recon_len` history steps to the horizon loss, as the official training loop does.
- With `use_marks`, the four hourly `timeF` calendar features become extra tokens in attention, dropped before the heads.

## When to use

- Multivariate data with many variates where quadratic channel attention is too expensive (linear-cost Dot-attention).
- Series with a clear trend plus seasonal remainder, handled by separate paths.
- Hourly data benefits from calendar tokens; for other sampling rates turn marks off.
- Variates are mixed through attention, so weakly correlated channels gain little from the channel path.

## Configure

- `enc_in`: number of data channels (variate tokens).
- `moving_avg`: odd moving-average window for the trend split (official preset 25); match it to the dominant period scale.
- `freq` / `use_marks`: only `h` can be rebuilt from raw marks; set `use_marks = false` for other sampling rates.
- `recon_len`: history steps reconstructed in the training loss, in `[0, seq_len]`; `0` gives the paper's horizon-only loss.

Other hyperparameters: preset defaults in `configs/models/Ister.toml`; tune generically.

## Differences

- CD_Ister variant (camera-ready per the official README); MP_Ister is not carried.
- `gelu_scores = true` (default) keeps the official GELU on `Softmax(Q) ⊙ K`; `false` gives Eq. (1).
- Training loss includes the 48-step history reconstruction like the official default branch; `recon_len = 0` is the paper's loss.
- Non-affine RevIN as in the code; the paper's unspecified MLP layouts follow the code.
- The official test path's `is_test=True` argument (a `TypeError` as published) is not reproduced.
