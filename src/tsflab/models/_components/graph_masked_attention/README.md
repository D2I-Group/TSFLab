---
name: "graph_masked_attention"
description: "Multi-head attention averaging a dense softmax with a softmax restricted to adjacency-permitted pairs; plain attention without a mask. Use for node attention softly biased by a known graph; not for hard graph-restricted attention, very many tokens (dense O(L^2)), or rows with no allowed key and no self-loop."
---

# graph_masked_attention

## What it does

Scaled dot-product multi-head attention in which a fixed adjacency biases but does
not replace data-driven attention. With per-head
`scores = (Q K^T) / sqrt(head_dim)`:

```
global = softmax(scores)
local  = softmax(scores.masked_fill(~adj_mask, -inf))
attn   = (global + local) / 2          # adj_mask given
attn   = global                        # adj_mask is None
out    = out_proj(merge_heads(attn @ V))
```

## When to use

Use to let a fixed graph bias attention over nodes while keeping long-range learned
affinity. Do not use for hard graph-restricted attention (use an additive `-inf` mask
with a standard attention layer, e.g. `self_attention_family` with a mask from
`masking`), for large `L` (attention is dense, `O(L^2)`; see `node_visibility` for
subgraph grouping), or when a row with no allowed key should still attend (such
rows output only the `out_proj` bias unless self-loops are added).

## Interface

`GlobalLocalGraphAttention(model_dim, num_heads=8)`

- `model_dim` (int >= 1): total width; raises `ValueError` unless divisible by
  `num_heads`. `num_heads` (int >= 1, default 8): head count;
  `head_dim = model_dim // num_heads`. `num_heads < 1` raises `ValueError`.
- `forward(query, key, value, adj_mask=None)`: inputs `[..., L, model_dim]` with
  matching leading batch axes (for `extralonger`: `[B, T, N, D]` style layouts where
  the attended axis is second-to-last). Only the last two axes are attended. Heads are
  inserted before the length axis, so `adj_mask` must broadcast to
  `[..., num_heads, L_q, L_kv]` (a `[L_q, L_kv]` mask works). It is cast to bool on
  the score device; `True` marks an allowed pair. A `[B, T, L_q, L_kv]` mask for
  `[B, T, L, D]` inputs does not broadcast correctly; insert a head axis
  (`mask[:, :, None]`). The module docstring says "broadcastable to
  `(..., L_q, L_kv)`", which omits the head axis; this card follows the code.
- Inputs must be floating tensors on one device; `forward` accepts non-bool masks
  (cast with `.to(torch.bool)`, so any nonzero value is allowed).
- Returns `[..., L_q, model_dim]`.
- Parameters (state-dict keys): `fc_q`, `fc_k`, `fc_v`, `out_proj`, each `nn.Linear(model_dim, model_dim)`
  with `.weight`/`.bias`. No buffers; no dropout; stateless.
- Fully masked rows: a query row with no allowed key (an all-`False` mask row) gets
  an all-zero attention row (neither the dense nor the local term is applied), so its
  output is the `out_proj` bias; outputs and gradients stay finite. Rows with at
  least one allowed key are bit-identical to the plain formula. (`extralonger` ORs
  in the identity, so it never hits this case.)
- Quirk: the mask only changes the local half, so masked pairs still receive half of
  their dense probability; the layer is a soft bias, not a hard mask.
