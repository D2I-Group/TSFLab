---
name: "SORMamba"
description: "Inverted channel-token model where one shared conv-free Mamba scans channels in both orders and an MSE regularizer aligns the two scans, removing dependence on channel order. Use for multivariate data with cross-channel dependence and no natural channel order; not for probabilistic output or channel-independent data."
---

# SORMamba

## Idea

- `Model.embedding` turns each channel's whole look-back window into one token (Algorithm 1, line 1); inputs are standardized per window and restored on the output (`use_norm`).
- `CDMambaBlock` is the `mamba` component's `MambaBlock` with `use_conv=False`: the 1D convolution before the selective SSM is removed because channels have no sequential order (Sec. 4.1, App. D).
- `SORMambaLayer.scans` applies the same block to the channel tokens and to the reversed channel tokens (`z1`, `z2`, the second flipped back); the layer adds `z1 + z2` to its input, then `LN(MLP(LN(Z)))` (Algorithm 1, lines 3-6).
- `forecast_with_regularizer` returns the forecast from `Linear(d -> H)` per token and `sum_i MSE(z1^(i), z2^(i))`; `training_objective` adds `reg_lambda` times that sum to the forecasting loss (Eqs. 3-4).

## When to use

- Multivariate data where channels interact: the Mamba scan runs across channel tokens (S-Mamba style).
- Channels have no meaningful order: the bidirectional scan plus regularizer makes the result robust to how channels are listed.
- Time is handled by one linear embedding of the whole window per channel. Point forecasts only.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly.
- Other hyperparameters: preset defaults in `configs/models/SORMamba.toml` (catalog S-Mamba widths, `reg_lambda = 0.01`); tune generically (the paper grid-searches `lambda` over {0.001, 0.01, 0.1}).

## Differences

Independent implementation from arXiv 2410.23356v1; the linked repository `seunghan96/SOR-Mamba` is unavailable (404), so there is no pinned revision, license, or reference comparison.

- The reversed scan is flipped back before the sum (Algorithm 1 omits this; the figures draw it).
- `LN(MLP(LN(Z)))` without residual by default (Algorithm 1); `mlp_residual = true` gives the S-Mamba form.
- The SiLU before the SSM is removed with the convolution; `reference_dt_init=True`, `d_inner = expand * d_model`, `dt_rank = ceil(d_model / 16)` as in the catalog S-Mamba.
- Unstated hyperparameters use catalog S-Mamba values (`d_model = d_ff = 128`, `d_state = 16`, two layers, `expand = 1`, dropout 0.1, GELU).
- CCM self-supervised pretraining, linear probing, and transfer settings are not provided (this is the supervised variant of Table 2). The regularizer applies in training only; validation and test use the configured criterion. Reported benchmark numbers are not reproduction claims.

Citation: Lee, Hong, Lee, Park, "Sequential Order-Robust Mamba for Time Series Forecasting", NeurIPS 2024 Workshop on Time Series in the Age of Large Models, arXiv:2410.23356.
