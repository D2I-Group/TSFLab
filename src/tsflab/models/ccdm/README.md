---
name: "CCDM"
description: "Conditional DDPM with channel-independent MLP encoders, a channel-token DiT denoiser, and a denoising-error InfoNCE on augmented negatives; returns sample quantiles. Use for probabilistic multivariate forecasting with correlated channels; not for point-only, low-latency inference."
---

# CCDM

## Idea

- Conditional DDPM: history and target are normalized by the history's per-channel mean and std; samples are mapped back.
- The denoiser embeds history and noisy target per channel with shared TiDE-style residual MLPs, then mixes one token per channel with adaLN DiT blocks (self-attention over channels, conditioned on the diffusion step).
- Training adds to the noise-regression loss an InfoNCE that scores the true target against patch-shuffled and amplitude-scaled negatives by denoising error (Eq. 3-6).
- `forward` runs `num_samples` ancestral trajectories and returns empirical quantiles `[B, H, D, Q]`.

## When to use

- Probabilistic multivariate forecasting where quantiles or samples are needed and channels interact (attention across channel tokens).
- Costly: training runs `1 + 2 * n_negatives` denoiser passes per sample, inference `num_samples * diffusion_steps`; not for tight compute budgets.
- Calendar marks are ignored.

## Configure

- `enc_in`: the dataset's channel count (one token per channel).
- `shuffle_patch_len`: `pred_len` must be at least twice it, otherwise patch-shuffle negatives equal the target.

Other hyperparameters: preset defaults in `configs/models/CCDM.toml`; tune generically.

## Differences

- Independent rewrite from the paper after reading the pinned official code (no license file); nothing copied.
- Eq. 3 scoring and one shared contrastive noise follow the paper (`contrast_form = "similarity"` reproduces the code's cosine scoring); the one-CiDM decoder follows the code.
- Point metrics use the runner's reduction of the quantile output (the paper reports the sample median).
- The configured observation loss only drives validation; the optional two-stage pretrain/fine-tune schedule is not built in.
- Full detail in reference.md.
