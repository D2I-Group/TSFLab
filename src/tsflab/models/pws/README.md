---
name: "PWS"
description: "In-repo TSFLab baseline: folds the lookback by period and, per phase patch, maps past periods to future periods with a residual MLP plus a linear weighted sum. Use as a simple periodic reference on seasonal data; not for aperiodic series or cross-channel modelling."
---

# PWS

## Idea

- The lookback is rearranged so each position within a `period` becomes a series over past periods (only the last `floor(seq_len / period)` full periods are used).
- Positions are grouped into patches of `patch_size`; each patch has its own residual `analysis_layers` MLP (default hidden 512, 256) over the period axis: `A_k(X_k) + X_k`.
- A per-patch `nn.Linear` (`weighted_sum_layers`) maps past periods to future periods, `Y_k = W_k(A_k(X_k) + X_k) + b_k`; the output is unfolded and trimmed to `pred_len`.
- No attention or convolution; optional `revin` normalization handles distribution shift.

## When to use

- Strongly seasonal data with a known period: the forecast for each phase is a learned combination of the same phase in past cycles.
- A cheap periodic baseline for comparing more complex period-aware models.
- Not for aperiodic series; channels are independent with shared weights.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.
- `period` follows the dataset period: must lie in `[1, seq_len]`; leftover steps beyond whole periods are dropped from the front of the lookback.
- `patch_size` follows `period`: must lie in `[1, period]`; patches are `ceil(period / patch_size)` phase groups.

Other hyperparameters: preset defaults in `configs/models/PWS.toml`; tune generically.

## Differences

Intentional TSFLab baseline with no external paper or official reference repository, hence no citation.

- `analysis_hidden` is a typed list and `analysis_act` accepts only implemented activations, preventing silent no-op configurations.
