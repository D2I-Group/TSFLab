---
name: "CrossLinear"
description: "Lightweight linear forecaster with a time-invariant cross-variate convolution embedding, patch projection, and a global linear head. Use for multivariate or exogenous-variable data with stable direct dependencies under tight compute; not for time-varying cross-variable relations."
---

# CrossLinear

## Idea

- `CrossCorrelationEmbedding` blends the input with one Conv1d across variables using a learned alpha, capturing time-invariant direct dependencies without a deep mixer.
- `PatchForecastHead` projects patches with a small MLP and blends them with positional embeddings by a learned beta.
- One global linear layer over all patch embeddings gives the forecast; non-affine `revin` wraps the model.
- Weights are shared across variables (the paper's many-to-many extension).

## When to use

- Forecasting with exogenous or correlated variables whose dependence is stable over time; ignoring time-varying and indirect dependencies limits overfitting.
- Very cheap: one convolution and linear layers.
- Point forecasts only.

## Configure

- `enc_in`: the dataset's channel count.

Other hyperparameters: preset defaults in `configs/models/CrossLinear.toml`; tune generically (non-divisible lookbacks are zero-padded to whole patches).

## Differences

- Clean-room implementation of paper Eqs. (3)-(11); the MIT official repository was a reference only.
- Implements the weight-shared many-to-many extension, not the target-channel many-to-one (MS) data path.
- The publication's data pipeline and optimization protocol are not reproduced.
