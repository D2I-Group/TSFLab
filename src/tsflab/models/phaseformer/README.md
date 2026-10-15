---
name: "PhaseFormer"
description: "Tiny periodicity model: tokenizes the lookback into one token per phase of the dominant period and mixes phases through a few learned routers. Use for efficient forecasting of strongly periodic series, including large channel sets; not for aperiodic data or when the period is unknown."
---

# PhaseFormer

## Idea

- `_tokenize` reshapes the lookback into `period` phase tokens (default 24), each embedding that phase across past cycles (`[phase, period-index]`).
- `CrossPhaseRouter`: a small set of learned router vectors first attend to all phases (aggregate) and are then attended by them (distribute), avoiding full phase-to-phase attention.
- One shared linear `predictor` maps each phase token to the future cycles of that phase; outputs are interleaved back into time order.
- Channels share weights; `revin` (affine) wraps the model.

## When to use

- Series with a strong, stable dominant period: every token is one phase, so the model only works when values at the same phase of successive cycles are related.
- Tight compute or parameter budgets: the paper reports about 1k parameters and stresses large, complex datasets where comparably cheap models struggle.
- Not for aperiodic series or a wrong period; channels are processed independently, so cross-channel dependence is not modelled.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.
- `period` follows the dataset period (the paper estimates it by autocorrelation): the lookback is cut into `ceil(seq_len / period)` cycles (circularly padded when `seq_len` is not a multiple) and the forecast into `ceil(pred_len / period)` cycles, truncated to `pred_len`.

Other hyperparameters: preset defaults in `configs/models/PhaseFormer.toml`; tune generically.

## Differences

Clean-room rewrite of paper Eqs. (5)-(11), checked against `models/PhaseFormer.py` of the official MIT repository at the pinned revision; no source copied.

- `period` is an explicit configuration value; the paper estimates the dominant period by autocorrelation.
- The default uses one routing layer and channel-independent processing; it does not claim the paper's exact training recipe, period-selection pipeline, or the reported ~1k-parameter setting for every dataset.

Cite: Niu, Deng, Tong, "PhaseFormer: From Patches to Phases for Efficient and Effective Time Series Forecasting", ICLR 2026 (arXiv:2510.04134).
