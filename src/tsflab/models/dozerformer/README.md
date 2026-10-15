---
name: "Dozerformer"
description: "Patch Transformer whose attention keeps only Local, seasonal Stride and horizon-growing Vary pairs, plus a linear trend branch. Use for long-lookback multivariate forecasting with a clear seasonal period; not for data without a stable period or when real sparse-attention savings are needed."
---

# Dozerformer

## Idea

- Moving-average decomposition (kernels 13 and 17, averaged): a linear layer maps the trend history to the horizon; the seasonal residual goes through the Transformer.
- A `3 x 1` conv lifts each variable to `embed_dim` feature maps, cut into `ceil(L/p)` patches; every variable is a separate token sequence (channel-independent, shared weights).
- Self-attention scores only Local pairs (`|i - j| <= floor(w/2)`) and Stride pairs (`|i - j|` a multiple of `stride + 1` patches), i.e. neighbouring and same-season patches.
- Cross-attention adds a Vary window: the first horizon patch sees `vary_len` recent encoder patches, each later one sees one more.
- A `1 x 1` conv reduces decoder feature maps to the seasonal forecast, added to the trend forecast and RevIN-denormalized.

## When to use

- Designed for series with a dominant seasonal period that whole patches can align with, so Stride attention links same-phase patches.
- Suited to long lookbacks (official scripts use 720 steps with 24-step patches) where full attention would mostly attend to irrelevant history.
- Channel-independent: it does not model cross-channel dependence.
- The default score mode is dense in effect (as in the official code), so do not pick it for compute savings.

## Configure

- `enc_in`: number of channels of the dataset.
- `patch_size`: patch length in steps; choose so a seasonal period is a whole number of patches (24 for hourly data with a daily period).
- `stride`: the seasonal interval in patches minus one (`stride + 1` patches = one seasonal period); `0` disables Stride attention.
- `label_len`: decoder history length; must not exceed `seq_len`.

Other hyperparameters: preset defaults in `configs/models/Dozerformer.toml`; tune generically.

## Differences

Independent rewrite from arXiv 2312.06874 (v2) after reading GRYGY1215/Dozerformer at `7598cf97`; no source copied. Defaults follow `scripts/ETTh1.sh` and `run.py`.

- The decoder input is built from the last `label_len` history steps plus zeros; `x_dec` and time marks are ignored.
- `score_mode = "zero"` (default) reproduces the official dense-in-effect attention; `"masked"` gives the paper's sparse semantics.
- Cross-attention always uses the cross pattern (the official code switches on equal lengths).
- `trend_branch = false` reproduces the Exchange/ILI variant; `normalize_decoder_input` optionally fixes the un-normalized decoder input.
- No checkpoint or published-metric reproduction is claimed; full detail in `reference.md`.
