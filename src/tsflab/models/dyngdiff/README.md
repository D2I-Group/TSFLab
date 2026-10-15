---
name: "DynGDiff"
description: "Unconditional S4 diffusion model guided at sampling time by a policy network's per-variable, per-step precision on a quantile likelihood. Use for probabilistic multivariate forecasting with quantile output; not for cheap point forecasts (sampling runs the full reverse diffusion per trajectory)."
---

# DynGDiff

## Idea

- `S4NoiseBackbone` is an unconditional noise predictor over `[context; horizon]` windows built from bidirectional S4 residual blocks (full HiPPO-LegS NPLR kernel).
- Stage 1 (`Model.pretrain`): fit the backbone with the noise MSE (Eq. 2) on mean-absolute-scaled windows, then freeze it.
- Stage 2: `StateAwarePolicyNetwork` reads `[x_t ; x0_hat]` and predicts a score `Z_t`, regressed onto a clipped z-scored log-precision proxy (Eqs. 8-10).
- Sampling: `A_t = exp(Z_t) / mean exp(Z_t)` weights a pinball loss on the observed context; its gradient shifts the DDPM reverse mean (Theorem 1, Eq. 14), so guidance strength adapts per variable and step.
- `forward` returns empirical quantiles `[B, pred_len, enc_in, K]` of `num_samples` guided trajectories.

## When to use

- Designed for probabilistic forecasting: the output is a set of quantiles from sampled trajectories.
- State-aware guidance weights reliable context positions more, which targets series whose noise level varies across variables and time.
- Expensive: two training stages, and validation and inference run the full guided sampler; avoid under tight compute.
- Joint diffusion over all channels (channel-mixing); no covariates or calendar marks are used.

## Configure

- `enc_in`: number of channels (the diffusion is multivariate over all of them).

Other hyperparameters: preset defaults in `configs/models/DynGDiff.toml`; tune generically.

## Differences

Independent rewrite of Sections 3-5 and Appendices A-B (arXiv v1) after reading `TT-20011031/DynG-Diff` at `28ae9001` (no license file, recorded `NOASSERTION`); nothing copied. The S4 kernel is written from the S4 algorithm and checked against the dense state-space recurrence.

- Eq. 7 is applied once to the policy score (the official sampler applies it twice), and `Z_t`, not the normalized weight, is regressed (Eq. 10).
- The S4 kernel uses the conjugate-symmetric Cauchy sum (the official fallback depends on installed packages).
- Point metrics use the median of the quantiles; the paper uses the sample mean.
- Stage-1 epochs cycle the full training loader in float32 (official: a fixed cache of random windows, AMP).
- Validation uses the guided-sample quantile criterion; the paper selects nothing on validation data. Details in `reference.md`.
