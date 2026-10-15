---
name: "GBT"
description: "Two-stage Transformer: a frozen conv-attention pyramid gives a first forecast that ESM masked self-attention refines. Use for non-stationary multivariate series run channel-independently; not for calendar- or covariate-driven tasks (no marks) or probabilistic output."
---

# GBT

## Idea

- First stage (`AutoRegressionStage`, Sec. 4.1.2, Fig. 5b): `pyramid` branches read the last `ar_len // 2**i` steps through `ar_blocks - i` `ARBlock`s and a flatten-linear head that replaces cross-attention; branch forecasts are averaged.
- `ARBlock`: an encoder without feed-forward (weight-normalized point-wise attention, residual) followed by the Fig. 5a ConvBlock: Res-P (stride 2, max-pool shortcut) then Res-C (width doubling, point-wise-conv shortcut).
- `pretrain` trains only the first stage with MSE against the future window, then freezes it; the trainer's loop trains the second stage on the detached "Good Beginning".
- Error Score Modification (Eq. 3, Algorithm 1): a per-head, per-query scale `3 ** (sigmoid(5 x) + 1e-5) - 1` defines a Gaussian `N(j; 0, sigma_i^2)` over key positions, added to `QK^T` before the causal mask and softmax.
- Second stage: point-wise value embedding plus sinusoidal positions, `d_layers` post-norm decoder layers with masked ESM attention, a final LayerNorm and projection, added to the first-stage forecast.

## When to use

- Designed for non-stationary series, where a one-stage Transformer overfits shifting statistics; the frozen first stage supplies a stable starting forecast for the attention stage.
- Runs channels as independent units by default (`channel_independent = true`), as in the official multivariate scripts.
- Not for tasks that need calendar marks or exogenous inputs (marks are ignored), nor for probabilistic output.
- Needs a two-phase training budget (`ar_epochs` for the first stage, then the ordinary trainer).

## Configure

- `enc_in`: the dataset's channel count.
- `ar_len`: first-stage input length, in `[1, seq_len]` (default `seq_len`; the official ILI scripts use 18 of 36); `ar_len // 2**(pyramid - 1)` must be at least 1.

Other hyperparameters: preset defaults in `configs/models/GBT.toml`; tune generically.

## Differences

- Independent rewrite of Secs. 4.1-4.2 (Figs. 4-5, Eq. 3, Algorithm 1, Table 3) after reading the pinned official code; nothing copied. Underspecified details (embeddings, residual-unit layout, decoder norm, `mix` layout, channel independence, `ar_len`) follow the code.
- The ConvBlock follows Fig. 5a; the official fused stride-2 unit is `fused_conv = true`.
- The first stage is trained by `pretrain` for `ar_epochs` (default 1) without early stopping, then frozen with `requires_grad = False`; the second stage uses the TSFLab trainer.
- AR attention uses `n_heads` (officially 8); stride-2 units give `ceil(L / 2)` for any length.
- Not implemented: ProbSparse attention, the Autoformer format, RevIN (off in the scripts), calendar embeddings.

Full detail: `reference.md`.
