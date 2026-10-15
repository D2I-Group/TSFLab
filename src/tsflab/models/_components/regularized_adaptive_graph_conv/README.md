---
name: "regularized_adaptive_graph_conv"
description: "Row-swap regularizer for a node-embedding table plus a linear-in-nodes adaptive graph convolution over an implicit cosine adjacency E'E'^T (RAGC). Use for large node graphs where a dense N x N adjacency is too costly; not for explicit, directed, or per-sample dynamic graphs."
---

# regularized_adaptive_graph_conv

## What it does

Two pieces for a node-embedding adaptive-graph forecaster.

`StochasticSharedEmbedding(p)`: in training, each row of an `[N, C]` table is replaced
with probability `p` by a uniformly random row of the same table (full-row swap, not
element dropout); identity in eval or when `p == 0`.

`EfficientCosineGraphConv`: builds a support table
`E' = L2normalize_rows( softmax(relu(E Wg), dim=1) * relu(E Wf) )` (`[N, C]`) and
diffuses `x` over the implicit adjacency `A = E' E'^T` without forming it:
one hop is `D^{-1} E' (E'^T x)` with `D = E' (E'^T 1) + 1e-6`, in `O(N C H)`.
Output is `Linear(concat[x, hop_1(x), .., hop_order(x)])` (no bias) with optional dropout.

## When to use

Use for spatiotemporal data with many nodes (large road networks) where a dense
`[N, N]` adaptive adjacency is too costly, and when the node-embedding table
benefits from row-swap regularization. Do not use if you need the adjacency
explicitly (it is never formed), directed/asymmetric propagation (`E'E'^T` is
symmetric), per-sample dynamic graphs (one support for the batch), or
per-time-step `[B, N, T, H]` layouts without reshaping.

## Interface

`StochasticSharedEmbedding(p=0.1)`: `p` in `[0, 1]` else `ValueError`. `forward(embeddings
[N, C])` returns the same shape; no parameters or buffers; uses the global torch RNG and
`self.training`. Rows may be replaced by themselves.

`EfficientCosineGraphConv(hidden_dim, spatial_dim, order=1, dropout=0.0)`:

- `hidden_dim` (`H`), `spatial_dim` (`C`) >= 1 and `order` >= 1 else `ValueError`.
  `dropout` > 0 adds `nn.Dropout`, otherwise `Identity`.
- Parameters: `gate_weight [C, C]` and `filter_weight [C, C]` (both initialised
  `randn`, so not scaled by fan-in), `out_weight [H, (order + 1) * H]` (kaiming
  uniform, `a = sqrt(5)`). No buffers.
- `compute_support(node_embedding [N, C]) -> [N, C]`: raises `ValueError` if the last axis
  is not `spatial_dim`.
- `forward(x [B, N, H], node_embedding [N, C]) -> [B, N, H]`: raises `ValueError` if
  `x.shape[-1] != hidden_dim`. Stateless apart from parameters; the support is recomputed on
  every call, so gradients reach the embedding table.
