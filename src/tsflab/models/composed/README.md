---
name: "Composed"
description: "Slot-assignment recombination model: pick normalization, decomposition, temporal encoder, channel sharing and head from cataloged components. Use for ablations and automated research over component combinations on point forecasting; not for probabilistic heads or reproducing a published model."
---

# Composed

## Idea

- Parameters are the slot assignment `normalization`, `decomposition`, `temporal`, `channel`, `head`; the loss stays in `training.loss`.
- A pipeline runs norm, optional split (residual and trend for `series_decomposition`), one temporal encoder per branch, concatenation, `flatten_forecast_head`, and denorm.
- `channel` sets weight sharing: `independent`, `individual` (per-channel head and linear weights), or `mixing` (only with `mixer_block`).
- Invalid combinations are rejected at config validation with every reason listed (`check_assignment`).
- `tsf model compose spec.toml` validates a spec; `--write-config run.toml --dataset <name>` writes a runnable config, and `--register NAME` scaffolds a dedicated catalog entry (needed for quantile or Gaussian heads).

## When to use

- Testing which component choice matters on a dataset: `revin`/`last_value_center` target non-stationary levels, `series_decomposition` targets trend, and the temporal option sets capacity.
- Point output only; marks and decoder inputs are ignored; with `features = "MS"` only the last channel is returned.
- Not a reproduction of any paper; use the dedicated model for a published architecture.

## Configure

- `enc_in`: the dataset's channel count.
- `patch_len`: must not exceed `seq_len` for the patched options (`tst_transformer`, `mamba`).

Other hyperparameters: preset defaults in `configs/models/Composed.toml`; tune generically (`kernel_size` odd, `hidden` divisible by `n_heads`).

## Differences

- No paper and no official code: a TSFLab-originated mechanism, so the card carries no `code`, `revision` or `license`.
- Each slot option keeps the interface documented in its component card; their differences from their papers are recorded there.
- Options, rejected options and slot mechanics are listed in reference.md.
