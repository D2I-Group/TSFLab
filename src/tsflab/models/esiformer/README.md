---
name: "Esiformer"
description: "iTransformer-style variate tokens over a midpoint-interpolated window (2L-1 steps), with a post-norm encoder whose FFN uses a fixed random sparse mask. Use for long-term multivariate forecasting on hourly data with calendar features; not for irregular sampling or tasks needing channel independence."
---

# Esiformer

## Idea

- Interpolation (Section II-B): keep every step and insert the midpoint of each neighbouring pair (`two_aver`, `2L - 1` steps) or a four-point mean after every second step (`four_aver`, `3L/2 - 1` steps), per channel, before normalizing the window.
- Sparse FFN (Section II-C): `y = (W * M) x + b` with a fixed random mask `M` keeping `round((1 - sparsity) * in * out)` connections; `SparseEncoderLayer` replaces the dense FFN of a post-norm encoder block.
- Variate tokens come from `DataEmbedding_inverted` over the interpolated length; interpolated calendar features (TSLib `timeF`) are appended as extra tokens and dropped before the forecast.
- `X' = Interpolation(X)`, `H = Embedding(X')`, `H' = TransformerBlock(H)`, `Y = Projection(H')`, with Non-stationary-Transformer normalization around it.

## When to use

- Aimed at long-term multivariate forecasting where densifying the lookback and sparsifying the FFN regularize an inverted Transformer ("less is more").
- Attention over variate tokens mixes channels; cost grows quadratically with channel count.
- Calendar tokens assume hourly `timeF` features.

## Configure

- `enc_in`: number of channels.
- `interpolation = "four_aver"` needs an even `seq_len >= 4`; `two_aver` needs `seq_len >= 2`.
- `freq`: calendar-feature frequency; only hourly (`"h"`) is supported.

Other hyperparameters: preset defaults in `configs/models/Esiformer.toml`; tune generically.

## Differences

Independent rewrite after reading `yyg1282142265/Esiformer` at `02eee558` (no license file, recorded `NOASSERTION`); nothing copied.

- `four_aver` requires an even `seq_len` and never allocates uninitialised memory (official bug for odd lookbacks).
- One `interpolation` parameter sets both the transform and the embedding width; booleans are strict (official `type=bool` flags parse `False` as true).
- Only the static random mask is implemented (the official dynamic grow/prune path cannot run), and only the selected FFN is built.
- TSFLab trains with the configured run loss; the official code always uses L1 regardless of `--loss`.
- From the code, not the paper: `sparsity = 0.3`, uniform static mask, `d_model = d_ff = 128`, interpolation of the raw window before normalization, midpoint-interpolated marks.

Citation: Guo, Y., Zhao, Y., Dang, S., Zhou, T., Sun, L., Qian, Y. "Less is more: Embracing sparsity and interpolation with Esiformer for time series forecasting." arXiv:2410.05726 (2024).
