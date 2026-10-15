---
name: "RootPurge"
description: "Linear forecaster (frequency-domain or time-domain bias-free map on mean-centred windows) trained with a root-purging regularizer that penalizes the map's response to its own residual. Use for regularized linear long-term forecasting; not for nonlinear dynamics or cross-channel modelling."
---

# RootPurge

## Idea

- `linear_map` is the bias-free map `G_W` (Sec. 4.2): with `domain="frequency"` (the paper's main tables) an rFFT of the window, a complex `(L//2+1) -> ((L+H)//2+1)` bin map, an inverse rFFT to `L+H` scaled by `(L+H)/L`, keeping the last `H` steps; with `domain="time"` an `H x L` matrix. Shared by all channels unless `individual=true`.
- `forward` subtracts the window mean, applies `G_W` and adds the mean back.
- `pad_residual` is the operator `P` of Eq. (3): a horizon shorter than the lookback is placed in a zero window of length `L` at a random offset per sample and channel; a longer horizon is cropped to its first `L` steps.
- `purge_loss` (Eq. 3): the detached residual `Y_fut - G_W(Y_his)` goes through `G_W o P` (`purge_order` times) and its mean square is penalized with weight `purge_lambda * max(L/H, 1) ** purge_order`; `training_objective` adds it to the forecast loss.

## When to use

- Linear long-term forecasting where the plain least-squares map overfits: the purge term regularizes the map's characteristic roots so it does not respond to its own residual.
- Very cheap: one linear or spectral map per model (or per channel).
- Not for nonlinear dynamics or tasks needing cross-channel interaction.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.

No other data-dependent parameter (the purge weight is scaled by `max(seq_len / pred_len, 1)` automatically). Other hyperparameters: preset defaults in `configs/models/RootPurge.toml`; tune generically.

## Differences

Independent rewrite of Sec. 4.2 (Eq. 3, footnotes 3-4) after reading the official code (`Wangzzzzzzzz/RootPurge`, `63973fc`, no license file); nothing copied.

- The root-seeking term uses the run's configured criterion (MSE in the paper; set `training.loss = "mse"` to match).
- The inverse rFFT uses the explicit length `L+H` (official output is one sample short when `L+H` is odd).
- The official 5000-step Adam loop is replaced by the catalog trainer; the RRR and DWRR baselines of Sec. 4.1 are not included.
