---
name: "Extralonger"
description: "Traffic Transformer with parallel temporal, spatial (global-local graph attention) and mixed attention routes fused with fixed 1:1:2 weights, aimed at extra-long horizons. Use for node-graph traffic forecasting at horizons up to a week; not for data without a graph or calendar marks."
---

# Extralonger

## Idea

- Temporal route: compress the node axis into a per-step embedding, add time-of-day and day-of-week embeddings, and run self-attention over time.
- Spatial route: compress the time axis per node, append a learned node embedding, and attend over nodes with `GlobalLocalGraphAttention`, averaging dense and adjacency-masked softmax attention (`graph_masked_attention`).
- Mixed route: temporal attention, then a second attention pass with the feature axis as tokens.
- Fuse as `(temporal + spatial + 2 * mixed) / 4`; treating time and space in a unified way keeps cost low as the horizon grows.

## When to use

- Designed for long and extra-long traffic horizons (paper: 144-2016 steps, half a day to a week at 5-minute sampling) where separate spatial and temporal modelling becomes expensive.
- Needs a node graph (`adj_mx`) and calendar marks for time-of-day and day-of-week embeddings.
- Node-axis compression into one embedding per step suits correlated sensor networks; it is not channel-independent.

## Configure

- `enc_in`: number of nodes (fallback when the runner does not inject `num_nodes`).
- `adj_mx`: runner-injected dense adjacency; the mask is `(adj_mx != 0) | eye(N)`.
- `steps_per_day`: number of steps per day for the time-of-day table (288 at 5-minute sampling; the preset uses 24).

Other hyperparameters: preset defaults in `configs/models/Extralonger.toml`; tune generically.

## Differences

Local rewrite after reading the paper and `ZhuoLinLi-shu/Extralonger` at `4adb3ec1` (MIT; `model/Extralonger.py`, `model/config.yaml`, `train.py`); the three routes and fusion match the official code.

- The adjacency mask comes from the runner's dense `adj_mx` instead of the official `adj.csv` edge list (same symmetric self-looped semantics).
- The official learned `input_noise` input offset (not in the paper) is omitted.
- Multi-head splitting uses reshape instead of `cat(split(...))`; identical attention.
- The preset is smoke-scale (small widths, `seq_len = pred_len = 12`); the paper's extra-long-horizon results are not reproduced. Details in `reference.md`.
