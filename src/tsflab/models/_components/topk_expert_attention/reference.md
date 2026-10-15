# topk_expert_attention — reference

## Origin and granularity

Added with TimeExpert in the automated model intake (`d73cf9d0`); the only
consumer is `timeexpert`, which wraps it in `TMOEBlock` (attention plus
pre-norm feed-forward) over patch tokens. The cut contains the router, the
gather helper and the attention operator; patching, residual/FFN blocks, and
the head stay in the model. The `timeexpert` README (paper and pinned official code) is the
source for the method; this card makes no claim about official-code details beyond it.
`LocalExpertRouter` is not the same thing as `topk_expert_router` (see Related components).

## Invariants and equivalence evidence

- Contract checks (at extraction) of the local expert router and gather, and of the
  attention for `(topk, shared)` in `{(2, T), (2, F), (0, F), (0, T)}` and its errors,
  covered: router state dict empty, `[n, m, topk]` shapes,
  weights sum to 1, int64 indices in range, default scale, `gather_experts` rows equal to
  `w * kv[idx]`, `RuntimeError` for `topk > tokens`; attention state-dict keys, output
  shape and dtype, `last_route_weight` shape `[batch*heads, tokens, topk]` with unit sums
  (`None` and no `router` attribute when `topk == 0`), finite input and parameter gradients,
  and the `dim % num_heads` `ValueError`; a validation check covered the
  `num_heads`, negative `topk` and `qk_dim` `ValueError`s. Seeded outputs for
  `topk` 0 and 2, each plain and shared, and for the router were pinned as
  regression values.
- `timeexpert` structure checks: routing weights sum to one with `topk=2,
  shared=True`; `topk=0` falls back to full attention with finite output;
  forward/backward gives finite gradients for every parameter (including
  `qkv`, `positional`, `out_proj`) and a strict state-dict round trip.
- These checks passed in the full suite run of 2026-10-03 before the test suite was
  consolidated.
- Nothing compared against the official TimeExpert code; the pinned values above are
  regressions of this implementation only.

## Variants and options

`topk` (sparsity), `shared` (global mean expert), `qk_dim` (router/query-key
width), `dropout`. `topk=0` is the dense baseline.

## Related components

`self_attention_family` (dense and ProbSparse cores: ProbSparse keeps a subset of queries,
whereas here every query keeps its own top-k keys and re-weights them), `differential_attention`
and `global_patch_compression_attention` (dense, no routing), `topk_expert_router` (a gating
MLP plus top-k mixing of independent expert outputs, unlike this in-attention key/value
selection), `sparse_connection_router` (input-independent learned top-k connectivity, not
per-query), `soft_tree` (other differentiable routing).
- `graph_masked_attention`: sparsifies attention with a fixed graph rather than learned routing.
