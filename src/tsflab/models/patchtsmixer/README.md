---
name: "PatchTSMixer"
description: "Lightweight patch MLP-Mixer (TSMixer, KDD 2023): gated MLP mixing across patches, features, and optionally channels, with a shared linear head. Use for efficient long-horizon multivariate forecasting; not for exogenous inputs, probabilistic output, or self-supervised pretraining."
---

# PatchTSMixer

## Idea

- RevIN-normalized input (`revin`, no affine) is cut into overlapping patches and embedded by one shared linear layer, giving `[batch, channels, patches, d_model]`.
- Each `MixerLayer` applies `AxisMixer` blocks: pre-LayerNorm, an MLP along one axis (patches, hidden features, or channels), a `softmax_gate` gated-attention block, and a residual.
- `mode="common_channel"` (CI-TSMixer) shares all weights across channels and never mixes them; `mode="mix_channel"` (IC-TSMixer) adds an inter-channel mixer before the patch mixer in every layer.
- A flatten-linear head shared across channels (`flatten_forecast_head`) projects to the horizon; RevIN restores the scale.

## When to use

- Long-horizon forecasting under tight compute or memory budgets: MLP mixing only, no attention.
- `common_channel` for channels that move largely independently; `mix_channel` when cross-channel correlation carries signal.
- Not when timestamps or covariates matter (marks are ignored), when quantiles or distributions are needed (MSE point output only), or when masked pretraining is wanted (not implemented).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count.
- `patch_len`, `stride`: follow `seq_len`; `seq_len` must exceed `patch_len`, and the patch count is `(seq_len - patch_len) // stride + 1` (latest samples kept).

Other hyperparameters: preset defaults in `configs/models/PatchTSMixer.toml`; tune generically.

## Differences

- Independent rewrite of the supervised forecasting path, checked against the paper and the Hugging Face `transformers` implementation (pinned `3693f8d`, Apache-2.0).
- The channel mixer applies the gate after the MLP (as patch and feature mixers do); Hugging Face gates before the MLP in the channel mixer only, so `mix_channel` differs there.
- Only `std` RevIN scaling and LayerNorm; no patch self-attention, positional encoding, masked pretraining, distribution heads, online reconciliation head, or hybrid channel modeling.
- Defaults differ from Hugging Face (`d_model=16`, dropouts 0.1, `patch_len=16`); `init_std=0.02` initialization is not applied. Full list in reference.md.
