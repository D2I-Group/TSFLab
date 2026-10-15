---
name: "FTP"
description: "Pure-MLP model fusing multiscale channel-independent and channel-mixed patch branches with a Channel Enhancement stream. Use for long-horizon multivariate forecasting with a moderate channel count; not for hundreds of channels (dense channel-mixing maps) or univariate data."
---

# FTP

## Idea

- `_GlobalLocalLevel` has a channel-independent (`ci`) and a channel-mixed (`cm`) path, each combining patch-level (local) and whole-window (global) linear features; levels are applied recursively with patch lengths `patch_unit * (i+1)`.
- `channel_enhancement` scores latent dimensions and channels and adds a dominant-channel summary back to every channel; the paper's sampling is replaced by the probability-weighted expectation.
- `_FTPEncoderLayer` fuses the CI, CM and CE streams linearly, concatenates the original embedding, and projects back to `seq_len`.
- An MLP head maps each channel to `pred_len`; `revin` wraps the model.

## When to use

- Designed for long-term multivariate forecasting where it is unclear whether channel independence or channel mixing helps: it runs both in parallel and fuses them.
- CE targets data where a few dominant channels drive the others.
- Not for many-channel data: the channel-mixed path uses dense `enc_in * seq_len -> enc_in * d_model` maps, so parameters grow quadratically with the channel count.
- Point forecasts only.

## Configure

- `enc_in`: the dataset's channel count (sets the size of the channel-mixed linear maps).
- `patch_unit`, `num_scales`, `stride`: patch lengths `patch_unit * (i+1)` are clipped to `seq_len` and the tail is replicate-padded, so no divisibility with `seq_len` is required.

Other hyperparameters: preset defaults in `configs/models/FTP.toml`; tune generically.

## Differences

- Clean-room rewrite of Algorithm 1 (recursive multiscale GLF-CI/GLF-CM branches, latent/channel scoring in CE, tri-stream linear fusion, original-embedding concatenation, MLP horizon head); the official `models/FTP.py` and `layers/FTP_EncDec.py` were inspected at the pinned revision, nothing copied.
- The paper samples dominant channels probabilistically; this point forecaster uses the probability-weighted expectation for deterministic execution.
- The preset is compact and does not claim dataset-specific tuned widths or depths.
