---
name: "GWNet"
description: "Graph WaveNet: stacked causal dilated gated convolutions with diffusion graph convolution over a given adjacency and a learned adaptive adjacency. Use for spatio-temporal sensor networks (traffic) with a full or incomplete graph; not for unstructured channels or long horizons."
---

# GWNet

## Idea

- `gated_dilated_conv` applies causal filter and gate convolutions with dilation `2**layer` per `WaveNetGraphLayer`, so the receptive field grows exponentially with depth.
- `DiffusionConv2d` propagates over three supports: forward and reverse random walks of the given adjacency (`graph_utils.adj_to_supports`) and a learned adaptive adjacency from source/target node embeddings (`adaptive_node_embedding_adjacency`).
- Each layer has a residual path with `BatchNorm2d` and emits a skip connection; skips are summed across layers.
- A ReLU/1x1-conv head maps the summed skip features at the last time step straight to the `pred_len` horizon (non-autoregressive).
- Input features are the value plus calendar channels (`input_dim = 3`: value, time of day, day of week).

## When to use

- Designed for spatio-temporal graph forecasting (METR-LA, PEMS-BAY in the paper) where the given graph may be incomplete or not reflect the true dependency: the adaptive adjacency recovers hidden spatial relations.
- Works without a predefined graph (identity static supports), relying on the adaptive graph.
- Uses calendar marks as extra inputs, so timestamps should carry signal.
- Not for unstructured multivariate data with independent channels, or long horizons (the head emits all steps from one feature map, preset 12-step setting).

## Configure

- `enc_in`: the dataset's node count (runner-injected `num_nodes` takes precedence).
- `adj_mx`: predefined graph, injected by the runner; identity when absent.
- `layers` (with `blocks`, `kernel_size`): the causal receptive field `1 + blocks * (2**layers - 1) * (kernel_size - 1)` should cover `seq_len`, since the head reads only the last step (7 steps with the preset).

Other hyperparameters: preset defaults in `configs/models/GWNet.toml`; tune generically.

## Differences

- Local rewrite after reviewing the paper and the pinned BasicTS code: causal dilated gated convolutions, forward/reverse random-walk supports, a learned adaptive graph, diffusion convolution (order 2), residual paths and accumulated skip forecasts are retained.
- `residual_channels` must equal `dilation_channels`; adaptive embedding width is `min(10, max(2, num_nodes))`.
- Citation: Wu, Z., Pan, S., Long, G., Jiang, J., Zhang, C. "Graph WaveNet for Deep Spatial-Temporal Graph Modeling." IJCAI 2019, pp. 1907-1913.
