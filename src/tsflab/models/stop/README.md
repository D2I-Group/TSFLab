---
name: "STOP"
description: "Spatio-temporal MLP that routes all node interaction through a few learnable context units, with trend/residual encoders and calendar prompts. Use for node-level traffic-style forecasting under spatio-temporal distribution shift; not for data without time-of-day structure or for probabilistic output."
---

# STOP

## Idea

- `SeriesDecomposition` splits each node's lookback; trend and residual are encoded by separate linear maps and concatenated with time-of-day and day-of-week prompt embeddings.
- `CentralizedInteraction` replaces node-to-node message passing: nodes aggregate into a small bank of learnable context units (`core`) and diffuse back, giving a shared context and a personalized remainder.
- A refine MLP and a second residual mixer form a spatial branch; temporal and spatial linear heads are summed.
- `environment_forecasts` produces perturbed-message forecasts for the paper's robust-optimization idea; `forward` does not call it.

## When to use

- Node-level spatio-temporal forecasting (traffic, sensors) where test-time shifts in temporal patterns or node sets are expected; no adjacency matrix is needed.
- Data with daily and weekly rhythm: calendar prompts come from time-of-day and day-of-week marks (zeros when marks are absent).
- Short lookback/horizon settings (contract 12 to 12); encoders and heads are single linear maps over time.
- Not for probabilistic output, or when the paper's DRO training is required (not wired into the trainer).

## Configure

- `enc_in`: number of nodes/channels; the input must be `[B, seq_len, enc_in]` exactly.
- `tod_size`: number of time-of-day prompt bins; at most the steps per day at the data frequency (default 24 = hourly bins).
- `model_dim + 2 * prompt_dim` must be divisible by `head`.

Other hyperparameters: preset defaults in `configs/models/STOP.toml`; tune generically.

## Differences

- Clean-room, paper-derived; the unlicensed official repository was used only as reference.
- Covers equations (4)-(20): ConAU centralized interaction and bounded GenPU-style environment forecasts.
- The worst-environment DRO selection is not wired: the official trainer optimizes the plain masked loss with its DRO variance terms commented out, so no `training_objective` is declared.
- OOD splits and alternating GenPU/DRO optimization are not part of the generic trainer.
