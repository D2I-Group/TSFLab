---
name: "StaTS"
description: "Diffusion forecaster with a learned, spectrally regularized noise schedule and a frequency-gated x0 denoiser, sampled by DDIM into empirical quantiles. Use for probabilistic multivariate forecasting needing quantiles; not for cheap point forecasts (two-stage training, 100 trajectories per window)."
---

# StaTS

## Idea

- `Model.beta` is the Spectral Trajectory Scheduler: one learnable `beta(t)` per step, initialized linearly from `1e-5` to `0.1` (T = 50), clamped to `[1e-6, 1 - 1e-6]`; `alpha_bar` is its cumulative product.
- `ConditionalGuidedModule` (Eqs. 15-20) is a deterministic spectral anchor: rFFT of the normalized history, per-frequency gate, contiguous-band complex gains, irFFT, `Linear(L, H)`. `SpectralConditionedDenoiser` (Eqs. 21-25) gates each variable's noisy horizon by the history's spectral distortion and applies FiLM-modulated SiLU layers; `FrequencyGuidedDenoiser` fuses both (Eq. 26) and predicts `x0`.
- `objective_terms` / `schedule_loss` form the STS objective of Eq. 13 (barrier, terminal spectral KL, flatness progression, smoothness and initial-step penalties, forecasting loss) with the App. C.6 weights.
- `pretrain` is stage I: `schedule_epochs` rounds alternating a denoiser epoch and a schedule epoch, then the schedule is frozen; `denoiser_loss` is stage II.
- `sample` runs deterministic DDIM (`eta = 0`) from `x_T ~ N(0, I)` with a freshly corrupted history each step; `forward` returns empirical quantiles of `num_samples` trajectories.

## When to use

- Tasks that need quantiles or a predictive distribution rather than a point forecast.
- Multivariate data with informative spectral structure in the history (the denoiser is conditioned on frequency content); diffusion runs jointly over all channels.
- Expensive: two-stage training and `num_samples` DDIM trajectories per window at inference. Calendar marks are ignored.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly.
- Other hyperparameters: preset defaults in `configs/models/StaTS.toml` (paper Sec. 4.1 / App. C.6: lookback 168, T = 50, k = 3, two bands); tune generically. `quantile_levels` defaults to the evaluation's levels.

## Differences

Independent rewrite of Sec. 3-4 and App. A-C after reading `zjt-gpu/StaTS@5f0b50e` (no license, `NOASSERTION`); nothing copied. Recorded in `card.toml` issues:

- STS weights follow App. C.6 and the paper's sum form, since the official stage-I code reads undefined weights and fails.
- Follows the code where it departs from the paper: clamped raw `beta` vector, one tanh-averaged gate per variable, deterministic DDIM, pooled spectral flatness, DC excluded from terminal mass, x0 loss after de-normalization.
- Output is `[B, pred_len, enc_in, K]` quantiles; point metrics use the median. Validation uses the configured quantile criterion, not CRPS on 100 samples.
- Stage-II optimizer state is not carried over from stage I; trajectories are drawn in chunks of `sample_batch_size`.

Full detail in `reference.md`.
