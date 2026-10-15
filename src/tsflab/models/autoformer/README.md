---
name: "Autoformer"
description: "Progressive series decomposition in every layer plus FFT auto-correlation attention aggregating top-k delays. Use for long-horizon forecasting of periodic series with trend and calendar effects; not for aperiodic noisy data or lightweight budgets."
---

# Autoformer

## Idea

- `AutoCorrelation` replaces dot-product attention: FFT correlation of query and key finds the top-k delays, and time-rolled values are aggregated with softmax weights (sub-series level dependencies).
- `series_decomposition` runs after every attention and feed-forward sublayer, so the encoder keeps seasonal parts and the decoder accumulates trends progressively; the scaffold is the cataloged `decomposition_encdec` with `AutoCorrelation` as self and cross mixer.
- The decoder starts from a seasonal initialization (history tail plus zeros) and a trend initialization (history tail plus series mean).
- `forecast_embedding` embeds values and calendar marks; the output is projected seasonal part plus accumulated trend.

## When to use

- Long-term forecasting (energy, traffic, weather, economics) where periodicity is clear: auto-correlation discovers dependencies through period-based delays.
- Series with trend plus seasonality, which the inner decomposition separates at every layer.
- Uses calendar marks through the embedding; mixes channels in the embedding.
- Weak on aperiodic or noise-dominated series, where delay discovery has little to find; heavy (`d_model = 512`) for small budgets.

## Configure

- `enc_in`: number of channels; `dec_in` and `c_out` must match it.
- `moving_avg`: odd decomposition window.

Other hyperparameters: preset defaults in `configs/models/Autoformer.toml`; tune generically. `d_model` must be divisible by `n_heads`.

## Differences

Clean-room implementation; the MIT official repository was reference only. `SeriesDecomposition` maps Eq. (1), decoder initialization Eq. (2), encoder/decoder layers Eqs. (3)-(4), and FFT delay aggregation Eqs. (5)-(6). Inputs are `[B, seq_len, enc_in]` with optional six-column marks; outputs are `[B, pred_len, c_out]`. This forecast-only rewrite uses linear cross-context resizing and makes no checkpoint, training-recipe, or published-metric comparison claim.
