---
name: "BiST"
description: "MLP forecaster with label-side residual correction: virtual-cluster context and adaptive diffusion refine a base. Use for lightweight spatiotemporal node forecasting (traffic, sensors) with daily patterns; not for long horizons or data with no node structure."
---

# BiST

## Idea

- `series_decomposition` splits stable and trend parts, projected by separate linear maps and concatenated with learned node, time-of-day and weekday prompts.
- A stack of `ResidualMLP` layers forms the forward representation; `base_head` produces the base forecast.
- `node_queries` and `cluster_keys` define soft virtual clusters; their context graph separates common and personalized parts, from which a residual branch is built.
- `_adaptive_kernel` diffuses the residual over a learned graph with node-wise alpha/beta; `correction_head` adds it to the base forecast.

## When to use

- Spatiotemporal prediction where input and label dynamics are not consistent (the residual correction models what the forward pass misses), under tight time and memory budgets.
- The node graph is learned (virtual clusters, adaptive diffusion); no adjacency is required.
- Uses time-of-day and weekday prompts, so calendar marks matter; short windows (contract 12 in, 12 out); point output only.

## Configure

- `enc_in`: number of nodes.
- `tod_size`: steps per day of the dataset.
- `kernel_size`: decomposition window.

Other hyperparameters: preset defaults in `configs/models/BiST.toml`; tune generically.

## Differences

Clean-room implementation: Eqs. 5-24 map to the local decomposition, prompt, forward representation, virtual-cluster residual, adaptive diffusion, and correction modules. The unlicensed author repository was used for paper discovery only; nothing copied. Published-metric and checkpoint comparisons are not claimed.
