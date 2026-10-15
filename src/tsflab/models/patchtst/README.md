---
name: "PatchTST"
description: "Channel-independent Transformer over overlapping patches of each series, with a flatten head and RevIN. Use for long-horizon multivariate forecasting with long lookbacks; not for tasks needing cross-channel interaction, exogenous inputs, or probabilistic output."
---

# PatchTST

## Idea

- Cuts each channel's lookback into overlapping patches (default length 16, stride 8) that become Transformer tokens, shortening attention and keeping local semantics.
- Folds channels into the batch axis so all channels share one encoder (channel independence).
- Flattens all patch tokens and maps them linearly to the horizon; reversible instance normalization (`revin`) wraps the model.

## When to use

- Long-horizon forecasting where a long lookback helps: patching cuts attention cost, so the model can attend longer history.
- Multivariate data whose channels are weakly related or heterogeneous; the shared per-channel encoder avoids overfitting spurious cross-channel links.
- Not when cross-channel interaction carries the signal (no channel mixing), when exogenous covariates or calendar marks matter (marks are ignored), or when quantiles or distributions are needed (point output only).

## Configure

- `enc_in`: number of input channels of the dataset; must equal the data's channel count.
- `patch_len`, `stride`: follow `seq_len`; `patch_len` must not exceed `seq_len` (plus `stride` with `padding_patch = "end"`), and the token count `(seq_len - patch_len) / stride + 1` (+1 with end padding) sets attention cost and head width.

Other hyperparameters: preset defaults in `configs/models/PatchTST.toml`; tune generically.

## Differences

- Clean-room implementation of the paper's overlapping-patch equation, channel-independent shared encoder, and flatten head; Apache-2.0 official code used for reference only.
- Self-supervised pretraining, transfer learning, and residual-attention accumulation are not included; marks and decoder inputs are accepted and ignored.
- Does not use the cataloged `patchtst.PatchTSTBackbone` (used by `quantile_patchtst`): positional-encoding init, encoder layer, and head-dropout order differ, so migrating would change initialization and `state_dict()` keys. Details in reference.md.
