---
name: "node_visibility"
description: "Token-axis helpers: random keep/drop mask, per-sample shuffle and its inverse, and zero-padded grouping into fixed-size subgraphs and back. Use for making node attention affordable on large networks during training (VisiFold); not for inference-time use, graph-aware grouping, or padding-free attention."
---

# node_visibility

## What it does

Reduces the cost of all-pairs attention over `L` nodes (or any tokens) during
training. The five functions compose into a pipeline on `[B, L, D]` tokens:

1. `random_mask_tokens`: keep `max(1, int(L * (1 - mask_ratio)))` random positions (truncated, so floating-point rounding can drop one extra token)
   (sorted, shared across the batch).
2. `shuffle_tokens`: independent random permutation per sample.
3. `group_into_subgraphs`: right zero-pad to a multiple of `subgraph_size` and fold
   groups into the batch axis, so attention only spans `subgraph_size` tokens.
4. (model-local attention/head on the grouped tokens.)
5. `ungroup_subgraphs` then `unshuffle_tokens` invert 3 and 2.

## When to use

Use when quadratic attention over many nodes (large road networks) is too
costly to train: random node dropping and random fixed-size subgraphs bound the
attended set. Do not use when the pipeline must also run at inference (nothing
here is gated on training; the caller must gate it), when node order or graph
structure matters inside a group (groups are random subsets and carry no
adjacency), or when padding tokens must not take part in attention.

## Interface

All five are module-level functions in `__all__`; no parameters, buffers or state. Random
sampling uses the global torch RNG unless a `generator` is passed.

- `random_mask_tokens(x, mask_ratio, generator=None) -> (kept, keep_indices)`:
  `x [B, L, D]`; `mask_ratio` in `[0, 1)` else `ValueError`. `keep_indices` is a 1-D
  long tensor `[keep_length]` of sorted positions shared by all batch rows. With
  `mask_ratio <= 0` returns `x` unchanged and `arange(L)`.
- `shuffle_tokens(x, generator=None) -> (shuffled, perm)`: `perm [B, L]` gives, per row,
  the source index placed at each output position.
- `unshuffle_tokens(x, perm) -> Tensor`: inverse via `argsort(perm)`.
- `group_into_subgraphs(x, subgraph_size) -> (grouped, num_groups, orig_length)`:
  `subgraph_size > 0` else `ValueError`. If `L <= subgraph_size`, returns `(x, 1, L)`
  unchanged; otherwise `num_groups = ceil(L / subgraph_size)` and
  `grouped [B * num_groups, subgraph_size, D]`, zero padding at the end of the token axis
  (padding tokens are attended to unless the caller masks them).
- `ungroup_subgraphs(x, num_groups, orig_length, subgraph_size) -> Tensor`: returns `x`
  unchanged when `orig_length <= subgraph_size`, else regroups and drops padding
  to `[B, orig_length, D']`; `D'` may differ from the grouping width.
