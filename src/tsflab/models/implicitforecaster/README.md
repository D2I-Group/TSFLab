---
name: "ImplicitForecaster"
description: "Implicit decoder that predicts amplitude and phase for a pool of frequencies and composes the horizon by inverse rFFT, on a compact channel-independent MLP encoder. Use for long-horizon forecasting of series made of several periodic waves; not for aperiodic, spiky or probabilistic tasks."
---

# ImplicitForecaster

## Idea

- Instead of predicting each future time point independently, the Implicit Forecaster (IF) predicts the constituent waves (frequency, amplitude, phase) of the forecast, inspired by decomposition forecasting.
- A channel-wise MLP `encoder` of the window is concatenated with the history rFFT magnitude (amplitude branch) or phase angle (phase branch) (Eq. 2).
- `amplitude_head` predicts non-negative amplitudes and `phase_sine_head` / `phase_cosine_head` predict continuous phase coordinates combined by `atan2` (Eqs. 5-6), for `frequency_pool // 2 + 1` bins.
- `torch.polar` and `irfft` compose the wave pool into a signal; the first `pred_len` steps are the forecast. `revin` wraps the model.

## When to use

- Designed for long-term forecasting of series made of several long- and short-term periodic dynamics, where a global wave view beats independent per-step outputs.
- The paper's main use is as a replacement decoding head for larger backbones; this card pairs it with a compact MLP encoder.
- Channel-independent with shared weights: no cross-channel modelling.
- Not for aperiodic or spiky signals that a small wave pool represents poorly; point output only.

## Configure

- `enc_in`: the dataset's channel count.
- `frequency_pool`: wave-pool length tied to the horizon; effective value is `max(pred_len, frequency_pool)`, default `2 * pred_len` (preset 192); the paper searches it per benchmark.

Other hyperparameters: preset defaults in `configs/models/ImplicitForecaster.toml`; tune generically.

## Differences

- `spectral_parameters()` follows Eqs. (2), (5), (6), checked against `models/IFT.py` of the official repository at the pinned revision; written for TSFLab, nothing copied.
- Standalone model with a compact temporal MLP encoder, one frequency pool and one direct crop; the paper's backbones, training recipes and per-benchmark frequency-pool search are not reproduced.
- Citation: Li, Luo, Wang, Li, Peng, Liu, Guo, Zhang, Gong. "Towards Accurate Time Series Forecasting via Implicit Decoding." NeurIPS 2025 (vol. 38).
