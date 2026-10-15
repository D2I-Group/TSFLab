---
name: "STAEformer"
description: "Vanilla Transformer over spatio-temporal adaptive embeddings, alternating attention along time and along nodes, no graph convolution. Use for traffic-style node forecasting with strong daily and weekly calendar patterns; not for non-spatial multivariate data or long lookbacks."
---

# STAEformer

## Idea

- Complicated spatio-temporal architectures show diminishing gains; a learnable spatio-temporal adaptive embedding lets a vanilla Transformer capture intrinsic spatio-temporal relations and chronology.
- Concatenates value, time-of-day, day-of-week, optional node and a learnable `adaptive_embedding` of shape (seq_len, nodes) into one token per node and step.
- Alternates pre-norm self-attention along time and along nodes (`AxisAttentionBlock`); spatial relations come from attention plus the adaptive embedding, not an adjacency matrix (`adj_mx` is ignored).
- Flattens the time axis and maps it to the horizon with one linear layer (`use_mixed_proj`), or a two-step projection when disabled.

## When to use

- Sensor networks (traffic speed or flow) with short input windows and strong time-of-day and day-of-week patterns.
- No predefined graph needed; spatial structure is learned through node attention and the adaptive embedding.
- Attention runs over all nodes at each step and all steps per node, so cost grows quickly with node count and window length. Point forecasts only.

## Configure

- `enc_in` follows the node count; it must equal it exactly (the adaptive embedding is `(seq_len, nodes)`).
- `steps_per_day` follows the sampling frequency (288 for 5-minute data, 24 for hourly); it sizes the time-of-day embedding.
- `input_dim` follows the loader's input features (value plus time-of-day and day-of-week).
- Other hyperparameters: preset defaults in `configs/models/STAEformer.toml`; tune generically (embedding width divisible by `num_heads`).

## Differences

Local rewrite after reviewing the paper and the pinned BasicTS implementation (`GestaltCogTeam/BasicTS@c218c07`, Apache-2.0).

- Value, calendar, optional spatial, and adaptive embeddings feed alternating temporal-axis and spatial-axis self-attention blocks, then a direct horizon projection, as in the reference.
- `adj_mx` is accepted and ignored.

Citation: Liu, Dong, Jiang, Deng, Deng, Chen, Song, "STAEformer: Spatio-Temporal Adaptive Embedding Makes Vanilla Transformer SOTA for Traffic Forecasting", CIKM 2023, arXiv:2308.10425.
