---
name: "STHD"
description: "Scalable Transformer for high-dimensional series: each target attends jointly over its own patches and those of its top-K most correlated series from the training split. Use for many-channel data where a few related series inform each target; not for low-dimensional or uncorrelated channels."
---

# STHD

## Idea

- `top_k_related` / `Model.fit_relations` is Relation Matrix Sparsity (Sec. 4.1): Pearson correlation on the training split, self excluded, each channel keeps its top `k` series; `training_setup` fits it once and stores it in the `related` buffer.
- `Model.group_index` stacks every target with its related series, so the batch carries one `(1 + K) x L` sample per target (the 2-D input of Sec. 4).
- `Model.embed_groups` is Eqs. (1)-(3): the `embed` patch embedding (end-replicated padding, bias-free projection, time sinusoid) plus `channel_position_table`, a base-5000 sinusoid over the target/related axis.
- The `transformer_encdec` encoder attends over all `(1 + K) * P` patch tokens at once (Eqs. 4-5); `flatten_forecast_head` maps the target's flattened patches to the horizon inside `revin`.

## When to use

- High-dimensional multivariate data (hundreds or thousands of series) where full channel attention is too costly but each series has a few strongly correlated partners.
- Relations are fixed from training-split correlations; relations that change over time are not tracked.
- Each target is processed as its own group, so compute scales with `channels x (1 + k)` patch sequences. Point forecasts only.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly.
- `k` follows the channel count: at most `enc_in - 1` (clipped for low-dimensional data); default 13 from the official crime script.
- `patch_len` / `stride` follow `seq_len`: `patch_len <= seq_len`; `P = floor((L - patch_len) / stride) + 2` patches.
- Other hyperparameters: preset defaults in `configs/models/STHD.toml`; tune generically (`d_model` divisible by `n_heads`).

## Differences

Independent rewrite of Sec. 4 after reading `xinzzzhou/ScalableTransformer4HighDimensionMTSF@97a36a5` (no license, `NOASSERTION`); nothing copied. Recorded in `card.toml` issues:

- ReIndex (Sec. 4.2) batching is not reproduced: a batch of `b` windows becomes `b x N` target groups.
- Dense exact Pearson correlation replaces DeepGraph (same values).
- `patch_len` and `stride` are honoured (official model ignores them and uses 12 / 6).
- The paper's target-first grouping is used (official loader scrambles membership for `M > 1`).

Full detail in `reference.md`.
