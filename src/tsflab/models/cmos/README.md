---
name: "CMoS"
description: "Super-light linear forecaster: a per-channel softmax mixture of K shared chunk-to-chunk correlation matrices, with optional periodic initialization. Use for periodic series under tight parameter or compute budgets; not for exploiting cross-channel dependence, covariates, or probabilistic output."
---

# CMoS

## Idea

- Splits the series into chunks and learns `num_map` shared chunk-by-chunk correlation matrices mapping input chunks to output chunks.
- A per-channel strided `Conv1d` summary feeds a shared `allocator`; its softmax mixes the K matrices for each channel (`CorrelationMixer`).
- Optional `period` initializes the first matrix with periodic peaks (Sec. 3.3).
- Wrapped by `revin`; no decoder or nonlinearity, hence about 1% of DLinear's parameters.

## When to use

- Periodic data where the future chunk is a mix of correlated past chunks, especially under tight compute or parameter budgets.
- Channels are forecast independently (the mixture weights are per channel); cross-channel dependence is not modelled.
- Point forecasts only.

## Configure

- `enc_in`: the dataset's channel count (one Conv1d summary per channel).
- `seg_size`: must divide both `seq_len` and `pred_len`.
- `period`: optional; the dominant seasonal period in steps (for example 24 on hourly data), a multiple of `seg_size`; leave unset without a justified periodic prior.

Other hyperparameters: preset defaults in `configs/models/CMoS.toml`; tune generically.

## Differences

- Local implementation; the official `model/CMoS/Model.py` (no license) was inspected at the pinned revision and nothing was copied.
- Maps paper Eqs. 3-5 to K shared chunk-correlation matrices, channel-specific convolutional summaries, and a shared softmax allocator; an earlier non-paper top-k router was removed.
- Official initialization details, dataset recipes, and numerical reference comparison are not claimed.
