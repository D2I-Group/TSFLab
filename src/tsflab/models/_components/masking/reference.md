# masking — reference

## Origin and granularity

The classes came with the initial repository commit (`0b1fbf2e`, `models/module/masking.py`)
and match the long-standing Informer / Time-Series-Library mask helpers.
They were kept as a shared component because `self_attention_family` needs them
for full attention (`TriangularCausalMask`) and ProbSparse attention
(`ProbMask`), which serve `transformer`, `informer` and `dualformer` through that
component. `LocalMask` has no consumer in `src` (only the contract test builds
it) and its origin is not recorded. No model imports `masking` directly, which is
why the generated block lists no model consumers. Mask use stays model-local:
which attention flags it, where `-inf` is applied, and padding/validity masks.

## Invariants and equivalence evidence

- Contract checks (at extraction): bool dtype and shapes; the causal mask equals
  `triu(ones, 1)` and does not require grad; every `ProbMask` row equals
  `arange(key_length) > row id` for the selected rows; `LocalMask` is masked
  exactly where `j > i or j < i - ceil(log2(length))` with an unmasked diagonal.
  A 5-position causal mask and the 8-position local mask (not `ProbMask`) were
  pinned as regression values.
- The `self_attention_family` contract checks applied `TriangularCausalMask`;
  `transformer`, `informer` and `dualformer` exercised `TriangularCausalMask` and
  `ProbMask` through the model contract checks (indirect coverage).

## Variants and options

Three classes for three masking patterns. Missing: padding masks, non-causal
block masks, and graph-distance masks (see `graph_masked_attention`).

## Related components

`self_attention_family` (the consumer; masks are passed as its `attn_mask`
objects), `transformer_encdec` (encoder/decoder blocks that carry those masks),
`graph_masked_attention` (graph-structure masks), `node_visibility` (random
token masking and subgraph grouping, a different mechanism). The differential
attention in `differential_attention` and the routed attention in
`topk_expert_attention` take no mask object.
