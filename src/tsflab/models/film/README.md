---
name: "FiLM"
description: "Legendre-memory projection of the history, low-rank complex Fourier filtering to remove noise, and a mixture of multiscale-history experts. Use for channel-independent long-term forecasting from noisy histories; not for cross-channel modelling or very short lookbacks."
---

# FiLM

## Idea

- `LegendreProjection` runs the translated-Legendre recurrence `C_t = A C_(t-1) + B x_t` (bilinear-discretized), compressing the history into polynomial coefficients.
- `LowRankFourierLayer` keeps the lowest rFFT modes of the coefficient trajectory and filters them with complex low-rank factors (`order`, `rank`), removing noise.
- The last filtered state is reconstructed on the horizon grid by Legendre basis functions.
- `FiLMExpert`s read the last `scale * pred_len` steps (default scales 1, 2, 4) and are combined by a learned mixture; `revin` wraps the model.

## When to use

- Designed for long-term forecasting where history must be preserved without overfitting to its noise.
- Experts look back several horizon lengths, so it benefits from a lookback of a few multiples of `pred_len`.
- Channel-independent with shared weights: no cross-channel interaction.

## Configure

- `enc_in`: number of channels.
- `multiscale`: expert windows are `min(seq_len, scale * pred_len)`; give `seq_len >= max(multiscale) * pred_len` for distinct experts.

Other hyperparameters: preset defaults in `configs/models/FiLM.toml`; tune generically.

## Differences

Clean-room implementation; `tianzhou2011/FiLM` at `2794355f` (MIT) is reference only, nothing copied.

- Implements the Legendre recurrence with torch-native bilinear discretization, lowest-mode Fourier selection, complex low-rank factors, reconstruction and the multiscale expert mixture.
- `order` and `rank` replace the former ambiguous `window_size`.
- Random high-mode selection, use as a plug-in for other backbones, official initialization, and checkpoint or published-metric comparison are omitted.

Citation: Zhou, T., Ma, Z., Wang, X., Wen, Q., Sun, L., Yao, T., Yin, W., Jin, R. "FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting." NeurIPS 2022. arXiv:2205.08897.
