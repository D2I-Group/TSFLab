---
name: "adaptive_node_embedding_adjacency"
description: "Dense row-stochastic adjacency learned from node embeddings, softmax(relu(E1 @ E2)), with a single-embedding self-similarity form. Use for spatiotemporal data with a small-to-moderate node count and no or an incomplete predefined graph; not for large N (O(N^2) memory) or sparse graphs."
---

# adaptive_node_embedding_adjacency

## What it does

Builds a learned dense graph over `N` nodes from node embeddings, with no
predefined graph needed:

- dual form: `A = softmax(relu(E1 @ E2), dim=-1)` with `E1: [N, d]`, `E2: [d, N]`;
- single form (`target is None`): `A = softmax(relu(E @ E^T), dim=-1)`, with the
  transpose taken over the last two axes so batched `[..., N, d]` embeddings work.

Each row of `A` is a distribution over the columns, so `A` can be used directly as a
random-walk style support (`A @ x`) or as a basis element in a polynomial expansion.

## When to use

Use for a small-to-moderate node count where an `N x N` dense adjacency is
affordable and a self-learned, row-normalized graph is wanted alongside or instead
of a predefined one. Do not use for large `N` (memory is `O(N^2)`; see
`regularized_adaptive_graph_conv`), when a sparse graph is required (see
`sparse_connection_router`), or when a different graph constructor such as MTGNN's
top-k asymmetric one is intended.

## Interface

`adaptive_node_embedding_adjacency(source, target=None)` (pure function, no
parameters, buffers, or state):

- `source`: float tensor `[..., nodes, dim]`.
- `target`: optional float tensor used exactly as supplied with no implicit
  transpose, so it must already be `[..., dim, nodes]` (e.g. `gwnet` stores its
  target table as `[dim, nodes]`). Omit it for the self-similarity form.
- Returns `[..., nodes, nodes]` (or `[..., nodes, target_nodes]` if the target has a
  different node count, as in `adamshyper`, which passes a `[dim, hyper_nodes]` target) in the dtype/device of the inputs; rows sum to 1.
- It does no validation: shape mismatches surface as torch matmul errors.
- Differentiable w.r.t. both embeddings. Rows whose scores are all non-positive
  become uniform (`relu` gives zeros), not zero rows.
