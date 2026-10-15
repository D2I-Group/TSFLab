---
name: "SARAF"
description: "Linear forecaster fused with training-set futures retrieved by Pearson and calendar similarity, tuned to dataset stationarity. Use for recurring patterns with calendar effects on drifting, non-stationary series; not for probabilistic output or very long training splits (retrieval cost grows with them)."
---

# SARAF

## Idea

- `Model.fit_database` (the spec's `training_setup`) stores the scaled training series and raw calendar marks; the database is every stride-1 `(x_i, y_i)` pair (Sec. 3.3.1), and dataset stationarity `s-bar` is the mean of `window_stationarity` (Eqs. 1-2: across-sub-window variation of per-channel means and standard deviations).
- `temporal_similarity` (Eq. 4) is the Pearson correlation of flattened `L x C` windows after subtracting each window's last value; `time_aligned_bonus` (Eq. 3: hour, weekday/weekend, month, minute kernels) is blended in by `time_aware_weight` (Eq. 5).
- `stochastic_mmr` keeps the best of a Top-M pool and samples the other `K - 1` from `softmax(MMR / 0.1)`; `lambda(s-bar)` (Eq. 9) and the Gaussian bandwidth `sigma(s-bar)` (Eqs. 10-11) make retrieval more diverse and smoother on less stationary data.
- `forward` averages a linear forecast of the last-value-offset window (Eq. 13) with a horizon-linear map of the weighted retrieved futures (Eqs. 12, 14), applies the final horizon projection (Eq. 15), and adds the last value back.

## When to use

- Series whose future resembles past training windows, especially with calendar structure (hour, weekday, month) carried in the timestamps.
- Non-stationary data: retrieval diversity and smoothing adapt to the measured stationarity of the training set.
- Needs a stride-1 sliding-window training dataset; the database is stored in the checkpoint (`T x (C + 6)` floats) and retrieval costs `O(B * N * seq_len * C)` per batch for `N` training windows, so very long training splits get expensive.
- Point forecasts only.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly.
- `freq` follows the sampling frequency (`h`, `t`, `15min`, `30min`, `d`, `w`, `m`, `s`); it sets steps per day, which switches the calendar kernels on `seq_len` vs one day.
- `n_sub_windows` follows `seq_len`: `seq_len // n_sub_windows` must be at least 2 (stationarity sub-windows).
- Also `top_k <= candidate_pool`. Other hyperparameters: preset defaults in `configs/models/SARAF.toml`; tune generically.

## Differences

Independent rewrite from the paper and the pinned official code (`ShiqiaoZhou/SARAF@7b49aa8a`, no license file, `NOASSERTION`). Where paper and code disagree the code is followed (recorded in `card.toml` issues):

- Stationarity scores are normalised by each window's own per-channel std (code), not the dataset-wide std of Eq. (1).
- Windows and stored futures are offset by their own last value before similarity; an extra `Linear(pred_len, pred_len)` maps the retrieved forecast before the 0.5/0.5 average.
- Calendar weights and MMR temperature (0.1), unspecified in the paper, come from `layers/Retrieval.py`; the minute kernel uses raw minutes with period 60 as in the paper.
- Fixed official bugs: leakage masking covers only the exact overlap span `|j - i| <= seq_len + pred_len - 1` (no edge clamping), a never-failing tuple assert became an explicit shape check, and per-sample stationarity is computed once.

Citation: Zhou, Schöner, Wu, Fouché, Wilson, Wang, "Stationarity-Aware Retrieval-Augmented Time Series Forecasting", KDD 2026, arXiv:2606.04135.
