---
name: "SegRNN"
description: "Channel-independent GRU over fixed-length segments of the lookback with parallel multi-step decoding from positional segment queries. Use for long-lookback, long-horizon forecasting under tight compute; not for exploiting cross-channel dependencies or probabilistic output."
---

# SegRNN

## Idea

- RNNs struggle in long-term forecasting because of the many recurrent iterations; SegRNN reduces them with segment-wise iterations and parallel multi-step forecasting (PMF).
- The lookback is cut into `seg_len` segments, each linearly embedded (`segment_projection`) and fed to a GRU as one step.
- Every future segment is decoded in one pass from relative-position and channel-position embeddings with the encoder's final state as initial state (no autoregression).
- `segment_decoder` maps each decoder output to `seg_len` values; channels are folded into the batch and share weights; `center_on_last_value` / `restore_last_value` handle level shift.

## When to use

- Long lookbacks and horizons where a step-wise RNN would be slow; the paper reports large runtime and memory savings over Transformers.
- Channels treated independently with shared weights: suits weakly correlated channels, not data whose signal lies in channel interactions.
- Last-value centering makes it robust to level offsets between windows. Point forecasts only.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly.
- `seg_len` follows `seq_len` and `pred_len`: the model uses `ceil(seq_len / seg_len)` and `ceil(pred_len / seg_len)` segments; non-multiples are replicate-padded (local extension), so prefer a common divisor.
- Other hyperparameters: preset defaults in `configs/models/SegRNN.toml`; tune generically (`d_model` must be even).

## Differences

Paper-driven local implementation; the official repository is reference-only and no source was copied or adapted.

- Histories are last-value centered, split into segments, projected, and encoded by a shared GRU; relative-horizon and channel identifiers form decoder inputs, and each future segment is decoded independently from the same encoded state (PMF, not autoregression).
- Boundary padding for lengths that are not multiples of `seg_len` is a local runtime extension.

Citation: Lin, Lin, Wu, Zhao, Mo, Zhang, "SegRNN: Segment Recurrent Neural Network for Long-Term Time-Series Forecasting", IEEE Internet of Things Journal 13(5), 2026, doi:10.1109/JIOT.2025.3647705 (arXiv:2308.11200).
