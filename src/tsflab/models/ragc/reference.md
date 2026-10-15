# RAGC — reference

## Differences in detail

The official repository (`wkq-wukaiqi/RAGC@ee9cbb112b31ca704971a6b1d1c8ffdce6412b95`) publishes no
LICENSE file (recorded as `NOASSERTION`). TSFLab rewrites RAGC locally after reading the paper and
inspecting `models/RAGC.py` and `main.py`; no source was copied.

- Kept operations, rewritten as the reusable `regularized_adaptive_graph_conv` component: Stochastic
  Shared Embedding (`StochasticSharedEmbedding`, matching the official `SSE` module's
  row-swap-by-probability behavior exactly) and the Efficient Cosine Operator's gated cosine support
  plus kernel-trick diffusion (`EfficientCosineGraphConv`, matching the official `cos_gconv`
  branch's `emb_w1`/`emb_w2` gate/filter and `LinearizedSematicGraphConvolutional`'s two-einsum
  degree-normalized hop).
- The per-block residual-difference wiring (`hidden = hidden - smoothed; skips += smoothed`) and the
  dual regression heads (`regression_layer` on the residual state, `regression_skip` on the
  accumulated skip) match the official `forward`.
- Laplacian loss: the official repository has one explicit graph-regularization term,
  `RAGC.laplacian_smoothing()`, a penalty `trace(E^T L E)` on the node embedding `E` against the
  static road-network adjacency `L`, added as `loss_mae + loss_lap * 0.001`, but gated behind
  `--lap_sm`, which none of the released per-dataset commands (SD, GBA, GLA, CA) enable; the paper's
  "regularized adaptive graph convolution" refers to SSE. It is exposed as
  `Model.graph_regularization_loss()`, computed from the runner-injected static `adj_mx` (identity
  when none is supplied) and the raw node embedding, and is not called by `forward` or the training
  loop.
- `num_nodes`/`adj_mx` are runner-injected, not user parameters (the STID/AGCRN convention).
- The official `softmax_gconv`/`geo_gconv` ablation branches (plain softmax adjacency, or the raw
  static adjacency, in place of ECO) are not implemented.
- Dropout inside `EfficientCosineGraphConv`'s output projection defaults to `0.0` (official `0.1`)
  to keep the forward pass deterministic.
- `configs/models/RAGC.toml` uses smoke-scale widths (`spatial_dim=16`, `embed_dim=16`,
  `temp_dim_tid=8`, `temp_dim_diw=8`, `num_layer=2`), versus the official per-dataset configs
  (`spatial_dim=64`, `num_layer=4-8`).
- The official `time_of_day_size` (`steps_per_day`, e.g. 96/288) is generalized to `num_time_in_day`
  (default 24) over the normalized `[0, 1)` time-in-day convention (as in STID); `day_of_week_size`
  defaults to 7 as in the official code.

## Paper

Efficient Traffic Forecasting on Large-Scale Road Network by Regularized Adaptive Graph Convolution
(arXiv 2506.07179, 2025-06). STGCNs scale quadratically with road-network size, and approximation,
compression or partitioning trade away efficiency or accuracy. RAGC adds the Efficient Cosine
Operator (graph convolution from cosine similarity of node embeddings, linear time) and a
regularized adaptive graph convolution framework combining Stochastic Shared Embedding with adaptive
graph convolution through a residual difference mechanism.
