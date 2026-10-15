---
name: "global_patch_compression_attention"
description: "Two-stage cross-patch attention: each variable's last patch queries all patches to form a summary token, then every patch attends to the summaries (Sensorformer). Use for patch forecasters over many correlated variables where full cross-patch attention is too costly; not for few variables or causal group ordering."
---

# global_patch_compression_attention

## What it does

For tokens `P [B, G, N, d]` (G groups such as variables, N patches each):

1. Stage 1 (compress): query `q_g = P[:, g, -1]` (the last patch of each group),
   keys/values all `G*N` patches flattened. `s = LN(q + MHA(q, P, P))`, then
   `s = LN(s + MLP(s))`: one summary ("sensor") token per group, `[B, G, d]`.
2. Stage 2 (broadcast): every patch attends to the `G` summaries,
   `r = LN(P + MHA(P, s, s))`, then `r = LN(r + MLP(r))`.

Cost `O(G^2 N d)` instead of `O(G^2 N^2 d)` for full cross-patch attention.

## When to use

Use for patch-token forecasters over many variables where full cross-patch
attention is too costly and every patch should see cross-variable context
through a bottleneck. Do not use when ordering of groups must be preserved
causally (no mask), when `G` is tiny (the bottleneck gains nothing), or if the
last patch is not a sensible summary query.

## Interface

`GlobalPatchCompressionAttention(d_model, n_heads, d_ff, dropout=0.0)`:
`d_model`, `n_heads`, `d_ff` positive ints, `d_model % n_heads == 0` (else
`ValueError`); `dropout` in `[0, 1]` (range enforced by torch, not here) applied in both `nn.MultiheadAttention`
and both MLPs (`Linear-GELU-Dropout-Linear`).

`forward(patches)`: float `[B, G, N, d_model]`; raises `ValueError` if
`ndim != 4`. Returns the same shape. Parameters/state-dict keys: `compress_attn.*`,
`broadcast_attn.*` (`nn.MultiheadAttention`, batch-first), `compress_mlp.*`,
`broadcast_mlp.*`, `norm_compress_attn`, `norm_compress_mlp`,
`norm_broadcast_attn`, `norm_broadcast_mlp`. Stateless, no masking, no
positional information, attention maps are discarded. Because the query
is the last patch, patch order within a group matters; `N >= 1` is required.
