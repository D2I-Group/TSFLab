---
name: "ARMD"
description: "Deterministic diffusion that slides a window from history to future; a linear devolution net, few-step DDIM forecast. Use for cheap channel-independent forecasting where the history is at least as long as the horizon; not for probabilistic output or cross-channel modelling."
---

# ARMD

## Idea

- The diffusion chain slides a length-T window (T = `pred_len`) over history then future, so it is deterministic and ends at the history window.
- The devolution network is one `nn.Linear` over time shared by channels, mixed with the input by a learnable per-step weight (Eq. 5).
- `training_objective` uses a weighted L1 between the true and predicted evolution trends (Eq. 3, 6, 7) at one random diffusion step, replacing the configured loss.
- `forward` starts from the last T history steps and applies deterministic DDIM updates with the predicted trend in place of noise, in `sampling_steps` steps (one by default).

## When to use

- Lightweight forecasting that reframes the future as a continuation of the sliding series; inference is one or two linear steps.
- Needs `seq_len >= pred_len`; only the last `pred_len` history steps are used, so a longer lookback adds nothing.
- Deterministic sampler: point forecasts only, despite the diffusion framing.
- Channel-independent (shared linear map over time).

## Configure

- `enc_in`: number of channels.
- `sampling_steps`: at most `pred_len`; requires `seq_len >= pred_len`.

Other hyperparameters: preset defaults in `configs/models/ARMD.toml`; tune generically.

## Differences

Independent implementation after checking the pinned official code (MIT) and Eq. 1-10.

- History is the last `pred_len` steps of a longer lookback (paper and code: history length = horizon).
- Diffusion steps and schedules equal `pred_len` (official code hard-codes 96).
- The training deviation perturbs only the network input; the official in-place noise also leaks into the target.
- The official per-step loss weight (absent from Eq. 7) is reproduced.
- Only the deterministic DDIM sampler is provided; no stochastic ancestral sampler, no EMA, standard scaling instead of `[-1, 1]` min-max.

Full detail in `reference.md`.
