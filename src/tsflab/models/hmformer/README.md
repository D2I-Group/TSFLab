---
name: "HMformer"
description: "Channel-independent patch Transformer with parallel branches at doubling patch scales, rotary attention and stride-two fine-to-coarse mixing; branch forecasts are summed. Use for series with multi-scale periodicities; not for channel-interaction modelling or very short lookbacks."
---

# HMformer

## Idea

- Each `ScaleBranch` patches every channel (patch length and stride double per scale), embeds patches with a `Conv1d`, and runs post-norm Transformer blocks with `RotarySelfAttention`.
- Latent width doubles from fine to coarse branches (scale-adaptive feature expansion, SAFE); a stride-two `Conv1d` (`cross_scale`) passes each branch's tokens to the next coarser branch (hierarchical cross-scale mixing).
- Each branch has its own flatten-and-linear predictor and the branch forecasts are summed (multi-branch complementary prediction, Eqs. 2-5).
- Channels are folded into the batch so all share weights; no instance normalization is used.

## When to use

- Designed for series with complex multi-scale periodicities, for long-term and short-term forecasting.
- Channel-independent: no cross-channel modelling.
- Needs a lookback long enough for the coarser scales (`patch_len * 2**s <= seq_len`); with short windows it degrades to fewer branches.
- No instance normalization inside the model, so it relies on the dataset scaling for level shifts.

## Configure

- `enc_in`: the dataset's channel count.
- `patch_len`: finest patch length; scale `s` is kept only if `patch_len * 2**s <= seq_len`.
- `num_scales`: requested scale count; scales that do not fit `seq_len` are dropped.

Other hyperparameters: preset defaults in `configs/models/HMformer.toml`; tune generically.

## Differences

- Independent implementation of Eqs. (2)-(5) from the AAAI paper; the official `Code/Long-term_Forecasting/models/HMformer.py` (no license file, `NOASSERTION`) was inspected at the pinned revision as reference only; nothing copied.
- The default preset is compact (`K = 3` scales, one block per branch) rather than the full paper sweep; branches that cannot fit the history are omitted.
- Training recipes and reported weights are not reproduced.
