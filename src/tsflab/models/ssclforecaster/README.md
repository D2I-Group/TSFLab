---
name: "SSCLForecaster"
description: "Channel-independent linear short-term branch plus TCN and multi-scale moving-average long-term branch, trained with a contrastive loss weighted by whole-series autocorrelation. Use for long-horizon forecasting with long-range correlations beyond the window; not for cross-channel dependencies."
---

# SSCLForecaster

## Idea

- `normalize` is the window normalization of Eq. (4) (last value by default; mean, RevIN-style, or moving-average decomposition as options); `short_linear` is the short-term branch of Eq. (5).
- `represent` is the long-term encoder of Eq. (6): per channel, the TSLib token/position/`timeF` embedding (`embed.DataEmbedding`) and a dilated residual TCN, giving the representation used by AutoCon.
- `long_term` is the multi-scale MA decoder of Eq. (7): a time-axis linear map, then per scale `k` an edge-padded moving average of length `k + 1`, a channel MLP, and a second moving average, summed over scales.
- `fit_autocorrelation` (training setup) computes the ACF per channel on the smoothed training series (Eqs. 1-2); `window_index` recovers each window's global position from its first raw timestamp.
- `autocon_loss`: `global_autocon` contrasts max-pooled window representations across the batch weighted by `|R_SS(|t_i - t_j|)|`, `local_autocon` contrasts `T // 3` random steps inside each window; `training_objective` adds `autocon_lambda` times their average to the criterion (Eq. 8).

## When to use

- Long-horizon forecasting where correlations extend beyond the input window (long periodicities visible in the training-series ACF).
- Channel-independent with shared weights: suits weakly correlated channels, not data whose signal lies in channel interactions.
- Needs distinct raw calendar marks for every window (positions are recovered from timestamps); point forecasts only.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly.
- `freq` / `use_marks` follow the sampling frequency: calendar features support hourly `timeF` only (`freq = "h"`); set `use_marks = false` for other rates (AutoCon itself still needs raw six-column marks).
- `seq_len` must be even and at least 6, and every entry of `scales` even (moving-average kernels are odd).
- Other hyperparameters: preset defaults in `configs/models/SSCLForecaster.toml` (ETTh1 horizon-96 script); tune generically.

## Differences

Independent rewrite of Sec. 3 and App. A of the paper after reading `junwoopark92/Self-Supervised-Contrastive-Forecsating@f394705` (MIT); nothing copied.

- Window positions come from the first raw timestamp of each window (the catalog batch has no start index), which requires distinct six-column calendar marks.
- Positive/negative selection follows the official code, not the paper's Eq. (3).
- Every channel row uses its own sample's marks (fixes the official tile-wise mark pairing).
- A batch of one window has no global term; the univariate channel-mixing `AutoConNet` variant and the unused representation head are not implemented.

Full detail in `reference.md`.
