---
name: "NBeats"
description: "Univariate deep MLP with doubly residual stacks that emit backcast and forecast through polynomial-trend, Fourier-seasonality, or learned bases. Use for series dominated by trend and seasonality, with interpretable components; not for cross-channel interaction, covariates, or probabilistic output."
---

# NBeats

## Idea

- `NBeatsBlock` is a four-layer ReLU MLP whose two theta heads are projected by a basis: `trend_basis` (polynomial), `seasonality_basis` (Fourier), or a learned generic matrix.
- Blocks are chained with backward residual links (input minus backcast) and forward links (partial forecasts are summed).
- Default stacks are trend, seasonality, then generic (`stack_types`, `thetas_dim`); `share_weights_in_stack` shares blocks within a stack.
- Each channel is a univariate series folded into the batch, so all channels share one network.

## When to use

- Series whose structure is mostly trend plus seasonality: the trend and seasonality stacks give interpretable decomposed forecasts.
- Multivariate data with weakly related or heterogeneous channels, handled channel-independently; the paper targets univariate point forecasting (M3, M4, TOURISM).
- Not when cross-channel interaction carries the signal, when covariates or calendar marks matter (ignored), or when quantiles are needed (point output only). There is no instance normalization, so level shifts rely on the dataset scaler.

## Configure

- `thetas_dim` (seasonality entry) or `nb_harmonics`: the Fourier basis holds `ceil(dim / 2)` harmonics, harmonic `k` making `k` cycles over the horizon (and over `seq_len` for the backcast); to represent a period `p` the seasonality dimension needs at least `2 * (pred_len / p + 1)`.

Other hyperparameters: preset defaults in `configs/models/NBeats.toml`; tune generically. `enc_in` is accepted but does not shape the weights.

## Differences

- Rewritten for TSFLab after checking the paper and the pinned MIT reference implementation (`philipperemy/n-beats`); no source copied.
- Generic blocks learn their bases, trend blocks use polynomial bases, and seasonality blocks use Fourier bases, with time normalized to `[0, 1)` per window.
- Without weight sharing, the last block's backcast head is frozen because its backcast is never used.
- Published benchmark reproduction is separate from this implementation.
