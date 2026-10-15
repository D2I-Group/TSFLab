---
name: "DSTAGNN"
description: "Temporal and spatial attention modulating Chebyshev graph convolution, plus kernel 3/5/7 gated temporal convolutions. Use for traffic-style node forecasting on a road or sensor graph with dynamic spatial dependence; not for data without a graph or for long multivariate horizons without spatial structure."
---

# DSTAGNN

## Idea

- `AxisAttention` applies multi-head attention over time for each node and over nodes for each step; the spatial attention matrices are kept dense.
- `DynamicChebyshevConvolution` scales each Chebyshev support (`chebyshev_supports`, order 3) by the per-sample spatial attention before propagation, so spatial relevance changes with the input.
- `MultiScaleGatedTemporalConvolution` fuses three gated convolutions with receptive fields 3, 5 and 7 for wide-range temporal dependence.
- Only the value stream is used (no timestamp marks).

## When to use

- Designed for traffic flow forecasting on a road network, where spatial-temporal dependencies between nodes change over time.
- Needs a node graph (`adj_mx`); without one it falls back to identity adjacency and loses its graph propagation.
- Dense node attention: cost grows quadratically with node count.

## Configure

- `enc_in`: number of nodes.
- `adj_mx`: node adjacency of shape `[enc_in, enc_in]` from the dataset graph; identity when missing.

Other hyperparameters: preset defaults in `configs/models/DSTAGNN.toml`; tune generically.

## Differences

Independent rewrite from the public paper description; neither the official repository (`SYLan2019/DSTAGNN` at `10da0e08`, no license file, recorded `NOASSERTION`) nor a prior CauAir-derived file was used as implementation source.

- The paper's data-derived pattern-aware adjacency (the dynamic spatial-temporal aware graph) and temporal-distance matrix are not reproduced; a supplied `adj_mx` provides Chebyshev supports instead.
- Residual-attention accumulation across blocks, the paper's preprocessing and its training objective are not reproduced.
- Timestamp marks are not consumed; attention stays dense.

Citation: Lan, S., Ma, Y., Huang, W., Wang, W., Yang, H., Li, P. "DSTAGNN: Dynamic Spatial-Temporal Aware Graph Neural Network for Traffic Flow Forecasting." ICML 2022, PMLR 162, pp. 11906-11917.
