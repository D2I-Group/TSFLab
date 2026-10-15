---
name: "TiRex"
description: "Channel-independent xLSTM-style scalar-memory recurrence over value/mask patch tokens that rolls its state through masked future patches and outputs quantiles. Use for probabilistic forecasts where state tracking over long horizons matters; not for zero-shot use (no pretrained weights) or point-only tasks."
---

# TiRex

## Idea

- Each channel's z-scored history becomes patch tokens of values plus an observation mask (`ResidualProjection`).
- Zero-valued, missing-masked tokens for the future patches let the recurrent state roll forward across the horizon.
- `ScalarLSTMBlock` stacks RMSNorm, a stabilised exponential-gated `ScalarMemory` recurrence, and a feed-forward residual.
- Future tokens are projected to per-step features and `QuantileHead` outputs quantiles (default 0.1 to 0.9).

## When to use

- Designed for forecasts that need quantiles; training must use the quantile loss.
- The paper motivates the recurrent state for long and short horizons alike, relying on state tracking rather than attention.
- Channels are processed independently with shared weights; no cross-channel modelling.
- Not a zero-shot model here: weights are randomly initialized and trained per dataset, so the paper's pretraining benefit on scarce data does not apply.

## Configure

- `enc_in`: must equal the dataset channel count (`MS` outputs only the last channel).
- `patch_len`: the lookback is left-padded to a multiple of `patch_len` and the horizon is covered by `ceil(pred_len / patch_len)` future patches; divisors of `seq_len` and `pred_len` avoid padding and truncation.

Other hyperparameters: preset defaults in `configs/models/TiRex.toml`; tune generically.

## Differences

- Randomly initialized; the released pretrained model, its scale, training data, augmentations, and optimized xLSTM kernels are not reproduced.
- The scalar-memory recurrence is a stabilized version written from the public xLSTM equations.
- Uses the shared monotone `quantile_head` to satisfy the repository's non-crossing quantile contract.
- CPM (training-time masking) is exposed for training integration but not applied during inference.
- `src/tirex/models/tirex.py` and `src/tirex/models/slstm/block.py` were inspected at the pinned revision; no source was copied.
