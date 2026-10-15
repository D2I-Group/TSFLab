---
name: "node_adaptive_graph_conv"
description: "Node-adaptive Chebyshev graph convolution (AGCRN, HimNet): per-node weights from node embeddings via shared banks, over the learned softmax(relu(E E^T)) graph. Use for node-structured data with heterogeneous nodes and no reliable adjacency; not for fixed physical graphs or very many nodes."
---

# node_adaptive_graph_conv

## What it does

A graph convolution whose filter is different for every node. With node
embeddings `E` (`[N, D]`, or `[B, N, D]` per sample), banks `W_bank [D, K, I, O]`
and `b_bank [D, O]`, and `x [B, N, I]`:

```
A      = softmax(relu(E E^T), dim=-1)                      # adaptive_node_embedding_adjacency
T_0 = I, T_1 = A, T_k = 2 A T_{k-1} - T_{k-2}              # k < K
W_n    = E_n W_bank  ([K, I, O]),   b_n = E_n b_bank  ([O])
out[b, n] = sum_k (T_k x[b])[n] @ W_n[k] + b_n
```

The same embeddings both define the graph and generate the parameters, so the
filter needs no predefined adjacency and has `D*K*I*O + D*O` parameters instead
of `N*K*I*O + N*O`.

## When to use

Use for spatiotemporal data (traffic sensors, stations) whose nodes behave
differently and whose adjacency is unknown or unreliable: the graph and a
per-node filter are both learned from node embeddings, at the cost of a dense
`O(N^2 K)` basis and `O(N K I O)` generated weights per call. Do not use with a
fixed physical graph (see `graph_utils`, `diffusion_conv`), for large `N` (see
`regularized_adaptive_graph_conv`), or with batched embeddings that do not vary
per sample (pass the `[N, D]` table instead).

## Interface

`NodeAdaptiveGraphConv(in_dim, out_dim, order, embed_dim)`

- `in_dim` `I`, `out_dim` `O`: feature widths; `order` `K`: number of Chebyshev
  terms `T_0 .. T_{K-1}` (`K = 1` is a node-local, per-node linear map);
  `embed_dim` `D`: embedding width. No argument validation.
- `forward(x, embeddings)`: `x` is `[B, N, I]`; `embeddings` is `[N, D]` (one
  table shared by the batch) or `[B, N, D]` (per sample). Returns `[B, N, O]`.
  Any other embedding rank raises `ValueError`.
- Parameters (state-dict keys, in order): `weight_bank` `[D, K, I, O]`
  (Xavier-uniform), `bias_bank` `[D, O]` (zeros). No buffers, no dropout, no
  recurrent state; the graph and basis are recomputed on every call.
- Device and dtype: the identity follows the embeddings in the `[N, D]` path and
  `x` in the `[B, N, D]` path (as in the two source models); pass matching dtypes.
