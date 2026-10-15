---
name: "CrossGNN"
description: "GNN over multi-resolution views: FFT-period pooling, a scale-aware temporal graph, and a signed learned variable graph, on a last-value-centered series. Use for noisy multivariate data with several periods and heterogeneous channel relations; not for few-channel data (needs 2 * tk channels) or a known external graph."
---

# CrossGNN

## Idea

- `AdaptiveMultiScaleIdentifier` picks the batch's dominant FFT periods and average-pools the series at each, concatenating the scales.
- `SparseCrossGraphLayer` learns a temporal adjacency over all scale nodes, keeping top-k neighbours per scale plus adjacent steps.
- A learned variable graph passes positive messages from the top-k most similar channels and negative ones from the least similar.
- Series are centered on the last value; features are collapsed per channel, resized to `seq_len`, and mapped to the horizon by a linear head.

## When to use

- Noisy multivariate series with several dominant periods, where coarser scales expose clearer trend.
- Channels with both similar and dissimilar partners (signed variable graph); needs at least `2 * tk` channels.
- Marks and external adjacency are not supported; point forecasts only.

## Configure

- `enc_in`: the dataset's channel count; must be at least `2 * tk`.
- `tk`: neighbours kept per channel (at least 2, at most `enc_in / 2`).

Other hyperparameters: preset defaults in `configs/models/CrossGNN.toml`; tune generically.

## Differences

- Local implementation; the official `models/CrossGNN.py` (no license) was inspected at the pinned revision; nothing copied.
- Softplus replaces ReLU graph scores as a smooth positive relaxation; the data-dependent multiscale length is interpolated before the fixed-shape head.
- Dense score construction does not reproduce the paper's linear-memory claim.
- Checkpoint reference comparison and published metrics are not claimed.
