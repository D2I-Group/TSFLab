---
name: "TQNet"
description: "Lightweight MLP forecaster whose cross-channel attention uses a phase-aligned learnable periodic table as the query over the normalized window. Use for multivariate data with a stable calendar cycle and correlated channels; not for data without a fixed period or with timestamps that do not give hour or weekday phase."
---

# TQNet

## Idea

- `PeriodicQueryBank` holds a learnable table indexed by calendar phase (`cycle`, hour or weekday from the marks); a window of it, one vector per lookback step, is the query.
- A single `nn.MultiheadAttention` over channels-as-tokens uses that query against the normalized lookback (keys and values), fusing a global periodic prior with per-sample observations.
- The attended result is added to the input, projected, passed through a small two-layer MLP with a residual, and mapped to the horizon under `revin`.

## When to use

- Designed for multivariate series with a stable periodic pattern shared across samples (successor to CycleNet).
- Captures inter-variable correlation through the channel attention, so it targets correlated channels.
- Small MLP backbone: suited to tight compute budgets.
- Needs calendar marks; without them the phase falls back to zero and the periodic prior is not aligned.

## Configure

- `enc_in`: must equal the dataset channel count.
- `cycle`: the dominant period in steps, read from the marks as hour (default 24), weekday (7), or weekday*24+hour (168); other values use the hour mark, so they only fit hourly data.
- `channel_aggre_heads`: `seq_len` must be divisible by it.

Other hyperparameters: preset defaults in `configs/models/TQNet.toml`; tune generically.

## Differences

- Phase is derived from the four-input `x_mark_enc` contract (the CycleNet convention in this catalog) instead of the official bespoke data loader; it equals the official `cycle_index`, the phase of the first forecast step.
- The official attention dropout is hard-coded to 0.5; here it is a separate `attn_dropout` parameter (default 0.5).
- The official ablation switches `use_tq` and `channel_aggre` are not reproduced; the full configuration always runs.
- `models/TQNet.py` and `data_provider/data_loader.py` were inspected at the pinned revision; no source was copied.

Paper: TQNet reuses CycleNet's (arXiv:2409.18479) periodically shifted learnable vectors as the query of a cross-variable attention whose keys and values come from the raw lookback (paper Section 3, Figure 2). Citation: Lin, Chen, Wu, Qiu, Lin, "Temporal Query Network for Efficient Multivariate Time Series Forecasting", ICML 2025.
