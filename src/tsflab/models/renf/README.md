---
name: "ReNF"
description: "MLP forecaster with Boosted Direct Output: a stack of blocks where each regenerates a growing prefix of the horizon from the window plus all earlier sub-forecasts, trained with block-wise time and frequency L1 supervision. Use for long-term forecasting; not for horizons not divisible by the block count."
---

# ReNF

## Idea

- `Model.sub_forecasts` is Boosted Direct Output (Definition 2.3, Eq. 3): sub-forecaster k receives `[X, Y_hat_1, ..., Y_hat_{k-1}]` concatenated along time and regenerates the first `k * H / N` steps, so each block is a complete shorter forecast and the stack implicitly combines earlier forecasts.
- `FirstBlock`, `AlphaBlock`, `BetaBlock` follow Fig. 3 (`Proj(Transform(Proj(Norm(Drop(X)))))` with a linear head per block); `BetaBlock` adds latent skips between blocks, ReNF-alpha detaches the fed-back sub-forecasts.
- `block_supervision_loss` (Eq. 4): `sum_n (gamma / n) * (alpha * |FFT(Y_hat_n) - FFT(Y_n)|_1 + (1 - alpha) * |Y_hat_n - Y_n|_1)` over all sub-forecasts, declared as the `training_objective`.
- `revin` normalizes the window and denormalizes every sub-forecast; an optional learnable `[T, V]` position table is added. The reported forecast is the last sub-forecast.

## When to use

- Long-term forecasting where direct multi-output MLPs work well and extra supervision on shorter horizons helps: every block is trained on its own horizon prefix.
- The frequency term of the loss penalizes spectral errors in addition to pointwise L1.
- Not when `pred_len` cannot be split into equal segments; the `instance` block norm needs at least two variates.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels (also the size of the `[T, V]` position table).
- `num_blocks` follows `pred_len`: must divide `pred_len`.
- `block_norm` follows the channel count: `"instance"` normalizes across variates and needs at least two channels.

Other hyperparameters: preset defaults in `configs/models/ReNF.toml`; tune generically.

## Differences

Independent rewrite of Sections 2.2-2.4 after reading the official MIT code (`c232ec5`); nothing copied.

- The beta variant does not construct at the pinned revision (a refactor broke `Decoder` arguments); arguments are forwarded as the earlier `CapDecoder` did (`6bad210`).
- Not implemented: trainer-level weight EMA, StepLR, gradient clipping at 10, and the oracle "optimal" post-combination (analysis-only).
- Catalog `revin` uses `sqrt(var + eps)` (official `std + eps`).
- Validation and test use the configured criterion on the final forecast.
