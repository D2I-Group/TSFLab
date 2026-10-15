# graph_masked_attention — reference

## Origin and granularity

Added in commit `6663e2e0` (automated intake of VisiFold, Extralonger, ST-SSDL,
RAGC) as a new shared component for Extralonger's spatial route; the module docstring
cites the paper equation `GLSAtt = (Softmax(alpha_local) + Softmax(alpha_global)) V / 2`.
It is cut at the attention layer including the Q/K/V and output projections. Kept
model-local in `extralonger`: building `adj_mask` from the dense adjacency
(`(dense != 0) | eye`), the residual, layer norm and feed-forward wrapper, the
temporal and mixed routes (which call this layer with `adj_mask=None`), and all
embeddings. The generalization to other graph forecasters is a claim of the module
docstring, not evidenced by a second consumer.

## Invariants and equivalence evidence

- Paper-equation check (an `extralonger` check using a `[5, 5]` mask with self-loops
  plus one symmetric edge) recomputed the
  equation by hand from the layer's own projections and compares both the masked
  output and the mask-free (plain dense attention) output.
- Numeric-fix checks: a fully masked row gives a finite, bias-only output (zero
  attention, not NaN) with finite input gradients; rows with visible keys are
  bit-identical to the hand-computed equation and unaffected by masking another row.
  These checks passed in the full suite run of 2026-10-03 before the test suite was consolidated.
- `adj_mask` was only exercised with a 2-D `[L, L]` mask; per-head, per-batch
  or 4-D masks and cross-attention (`L_q != L_kv`) were never checked.
- No stored reference or pre-refactor copy exists, because the component
  was created directly for `extralonger` rather than extracted from existing code.

## Variants and options

- `adj_mask=None`: plain multi-head attention (self- or cross-attention, since
  `L_q` and `L_kv` may differ).
- The 50/50 blend of the two softmaxes is fixed; there is no learnable mixing weight,
  temperature, or dropout.

## Related components

- `self_attention_family`: plain full/probabilistic attention layers; this layer
  differs by blending a second, adjacency-masked softmax and owning its projections.
- `masking`: builds hard attention masks; here the mask only reshapes half of the weights.
- `node_visibility`: cuts the node set before dense node attention (scalability, not
  graph bias).
- `adaptive_node_embedding_adjacency`, `graph_utils`: sources of adjacency
  (a dense adjacency must be thresholded to a boolean mask first).
- `sparse_connection_router`: learned sparse mixing instead of attention.
- `topk_expert_attention`: also sparsifies attention, but by learned top-k expert
  routing rather than a fixed graph.
- `periodic_alibi_bias`: additive periodic distance bias, versus an adjacency blend.
