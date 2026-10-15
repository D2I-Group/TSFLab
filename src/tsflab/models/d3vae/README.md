---
name: "D3VAE"
description: "Generative forecaster: coupled diffusion of input and target into an NVAE-style bidirectional VAE, energy-based one-step denoising, and total-correlation disentangling. Use for short, noisy series where augmentation by diffusion helps; not for calibrated distribution output or tight compute."
---

# D3VAE

## Idea

- A GRU plus token/position/calendar embeddings builds the representation; an NVAE-style bidirectional VAE with hierarchical latent groups generates the target.
- Training diffuses input and target jointly (coupled diffusion, Eq. 3-6) and optimizes `psi * NLL + lambda * DSM + gamma * TC + MSE` (Eq. 14).
- An energy network trained by denoising score matching cleans the generated forecast in one step: `Y_hat - sigma0^2 grad E(Y_hat)` (Eq. 11).
- A total-correlation penalty disentangles the latent groups.

## When to use

- Short or noisy training series (the official setup slices short datasets), where diffusion augmentation and denoising regularize a generative model.
- Returns a stochastic cleaned point forecast; `forecast_with_uncertainty` exposes a noise estimate but no distribution is declared.
- Several networks and losses per step; not for tight compute.

## Configure

- `enc_in`: the dataset's channel count.
- `freq`: the dataset's sampling frequency code; selects calendar tables (`"t"` adds quarter-hour).
- `num_blocks`: `pred_len` must be divisible by `2 ** num_blocks`.

Other hyperparameters: preset defaults in `configs/models/D3VAE.toml`; tune generically.

## Differences

- Independent PyTorch rewrite of the Apache-2.0 PaddlePaddle code; nothing copied.
- Paper over code where the port is broken: real Gaussian diffusion noise, DSM gradients reach the energy network, TC is minimized, the sampling scale matches the NLL.
- `seq_len != pred_len` is supported through a model-local time projection (the paper assumes equal lengths).
- The repository trainer replaces the official training setup. Full detail in reference.md.
