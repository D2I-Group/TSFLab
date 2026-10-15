---
name: "DiPELinear"
description: "Parameter-efficient linear forecaster: static frequency and time attention, then a per-frequency complex FFT mapping to the horizon, with low-rank expert sharing and a spectrum-weighted loss. Use for long-term forecasting with long lookbacks under tight budgets; not for cross-channel modelling or probabilistic output."
---

# DiPELinear

## Idea

- Static Frequential Attention filters the history in the frequency domain (`irfft(theta * rfft(x))`); Static Temporal Attention rescales each step.
- Independent Frequential Mapping (`fft_extrapolation_conv`) zero-pads to history plus horizon and gives each rfft bin its own complex weight and bias, an `O(L log L)` kernel as long as the series.
- Low-rank sharing: `num_experts` weight sets per module are mixed per channel by `softmax(R / tau)`, `tau` annealed from 30 to 1 over the first 10 epochs.
- SFALoss mixes time-domain MSE with a spectrum L1 weighted by the SFA map; it replaces the configured criterion during training.

## When to use

- Long-term forecasting with long lookbacks (the paper uses 720 steps) where a linear model with few parameters is wanted.
- Channels are independent apart from expert routing; point output only.

## Configure

- `enc_in`: the dataset's channel count.

Other hyperparameters: preset defaults in `configs/models/DiPELinear.toml`; tune generically (`num_experts`, `use_revin`, `loss_alpha` vary by dataset in the official scripts).

## Differences

- Local rewrite after inspecting the MIT official code; outputs and SFALoss match the official module on seeded weights to 1e-5.
- SFALoss normalizes by the mean absolute weight (paper L1 form), not the code's signed mean.
- Only the routed formulation; the temperature anneal runs inside the model and uses the final value in eval.
- Per-dataset hyperparameters and the 720-step lookback are not preset. Detail in reference.md.
