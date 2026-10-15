---
name: "COSA"
description: "Test-time output-space adapter: a context-conditioned linear residual with a bounded tanh gate corrects a frozen base forecast. Use for online adaptation of deployed forecasters under non-stationarity and distribution shift; not as a standalone forecaster or for running the streaming protocol itself."
---

# COSA

## Idea

- Corrects a frozen base forecast (last-value `channel_wise_linear` by default, or an external `base_forecast`) instead of retraining the backbone.
- `correct` implements `H = W[Y0 || C] + b`, `Y = Y0 + tanh(g) H`, with context `C` the latest observed values or revealed-target means.
- Only `residual` and the scalar `gate` are trainable (`adaptable_parameters`).

## When to use

- Deployed forecasters facing non-stationarity or distribution shift, with ground truth revealed shortly after prediction; the paper reports the largest gains at long horizons.
- Architecture-agnostic and cheap: a single linear layer and a gate on the output.
- The streaming buffer, delayed labels and adaptive learning rate are outside the model; without them it returns the base plus an untrained correction.

## Configure

- `enc_in`: the dataset's channel count.

Other hyperparameters: preset defaults in `configs/models/COSA.toml`; tune generically.

## Differences

- Local implementation; the official `tta/cosa.py` was inspected at the pinned revision and nothing was copied. `correct` implements the paper equation exactly.
- The paper's streaming buffer, delayed-label update loop, PAAS, CALR and repeated optimizer steps are not run.
- A frozen last-value base and latest-input context are self-contained fallbacks; callers can pass an external frozen forecast and revealed-label context.
- License conflict upstream (CC BY-NC-SA 4.0 file vs MIT badge) is recorded in `card.toml`.
