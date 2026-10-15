---
name: "Client"
description: "Cross-variable Transformer over raw lookback variate tokens plus a weighted shared linear trend forecast. Use for long-term multivariate forecasting where channels are correlated and trend matters; not for calendar-driven or probabilistic tasks."
---

# Client

## Idea

- Each variate's normalized lookback is a token directly: no embedding, no positional encoding, so `d_model = seq_len` (Sec. 3.1).
- Full attention across variates in a post-norm conv-FFN encoder; a linear projection maps each token to `pred_len` (Eq. 1-2).
- One `Linear(seq_len, pred_len)` along time, shared by all variates, captures trend channel-independently (Eq. 3).
- Forecast is `F_trans + w_lin * F_lin` with a learnable per-variate `w_lin` (Eq. 4), then affine `revin` is reversed.

## When to use

- Long-term multivariate forecasting where cross-variable dependence is informative and a linear trend carries much of the signal.
- Calendar marks are ignored; point forecasts only.

## Configure

- `enc_in`: the dataset's channel count.
- `n_heads`: must divide `seq_len`, because the token width is the lookback length.

Other hyperparameters: preset defaults in `configs/models/Client.toml`; tune generically.

## Differences

- Independent rewrite of Section 3 and Algorithm 1 after reading the MIT official code; nothing copied.
- From the official code: `d_model` is the lookback length; the linear branch reads the RevIN-normalized input; `w_lin` is per variate; attention scales by `1/sqrt(d_head)`, not Eq. 1's `1/sqrt(C)`.
- Per-dataset searched hyperparameters are not reproduced. Full detail in reference.md.
