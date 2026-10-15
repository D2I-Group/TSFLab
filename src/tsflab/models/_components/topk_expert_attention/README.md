---
name: "topk_expert_attention"
description: "Temporal Mix-of-Experts attention (TimeExpert): each query keeps its top-k key/value positions, softmax-reweighted, with an optional shared mean expert and a depthwise-conv residual. Use for patch-token encoders that want sparse attention; not for causal attention or memory savings."
---

# topk_expert_attention

## What it does

Self-attention in which each key/value position is a candidate "expert". Per
head, for query `q_i`: router logits `l_ij = scale * q_i . k_j`; keep the `topk`
largest, `w = softmax(top-k logits)`; gather `k_j, v_j` of the selected
positions and multiply both by `w_ij` (`gather_experts`); then ordinary softmax
attention over the `topk` gathered pairs, `out_i = sum_j softmax(scale q_i . (w k_j))_j (w v_j)`.
With `shared=True` one extra expert, the mean key and mean value over all
tokens, is appended to every query's set. `topk = 0` falls back to full softmax
attention (plus the shared expert if enabled). Before projection a depthwise
`Conv1d(k=3, padding=1)` over time is added residually to `x`.

## When to use

Use for patch-token encoders that want each token to attend to a few relevant
positions plus a global fallback token. Do not use for causal attention, for long
sequences to save memory (the router still forms the dense `[tokens, tokens]`
score matrix), or when `tokens < topk`.

## Interface

`LocalExpertRouter(qk_dim, topk, scale=None)` (no parameters; `qk_dim` only sets the default scale): `forward(query, key)` with
`[n, m, c]` tensors returns `(weight, index)` each `[n, m, topk]`;
`scale` defaults to `qk_dim ** -0.5`. `torch.topk` raises `RuntimeError` if
`topk > m`.

`gather_experts(index, weight, kv)`: `index, weight: [n, m, topk]`, `kv: [n, m, c]`
-> `[n, m, topk, c]`, selected rows already multiplied by `weight`.

`TopKExpertAttention(dim, num_heads=8, topk=4, shared=False, qk_dim=None, dropout=0.0)`:
`num_heads >= 1` and `dim % num_heads == 0` else `ValueError`; `qk_dim` defaults to `dim` (per-head qk width is
`qk_dim // num_heads`; a non-divisible `qk_dim` raises `ValueError`);
`topk >= 0` (`0` is the dense mode; a negative value raises `ValueError`; `topk <= tokens` is required at call time, else `RuntimeError` from `torch.topk`); `dropout` in `[0, 1]` is
applied to the attention weights after the softmax over the selected set. `forward(x)`: float `[batch, tokens, dim]`
-> same shape (dtype and device follow the parameters; no padding or mask). Parameters/state-dict keys: `positional.weight/bias`
(depthwise conv), `qkv.weight/bias` (`dim -> 2*qk_dim + dim`), `out_proj.weight/bias`;
no `router` parameters (none exist), and no router at all when `topk == 0`.

Statefulness: each forward with `topk > 0` stores the router weights in the
plain attribute `last_route_weight` `[batch*heads, tokens, topk]` (not in the
state dict); it keeps the autograd graph alive until overwritten. No masks,
non-causal. Router weights sum to 1 over `topk` (tested), while the shared
expert is added after routing so attention weights over the `topk+1` set are
re-softmaxed.
