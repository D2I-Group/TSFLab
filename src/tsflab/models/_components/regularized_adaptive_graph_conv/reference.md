# regularized_adaptive_graph_conv — reference

## Origin and granularity

Added as new shared code with the `ragc` port; no earlier copies were consolidated. The
docstring names the paper's "Efficient Cosine Operator" and a "stochastic shared
embedding" regularizer. The cut is these two modules because both operate on a bare
node-embedding table and carry no spatiotemporal assumptions beyond a node axis.
Kept model-local in `ragc`: the embedding table itself, the per-window embedding,
feed-forward blocks between graph convolutions, the number of layers, `sse_p`/`use_sse`,
and the auxiliary losses.

## Invariants and equivalence evidence

- Contract checks (at extraction): empty state dict for the regularizer, identity
  in eval or at `p=0`, every output row being an original row, gradient flow at
  `p=1`, `ValueError` for `p=1.5`; for the convolution, state-dict keys
  `gate_weight`, `filter_weight`, `out_weight`, `out_weight` shape `[H, (order + 1) H]`,
  support rows of unit L2 norm, output shape and dtype, the kernelized hop equal to the
  explicit `D^{-1} A x` (atol 1e-4), gradients to `x`, the embedding and all
  parameters, `ValueError` for a wrong `spatial_dim`, wrong `hidden_dim`, `order=0` and
  `hidden_dim=0`, and `dropout > 0` selecting `nn.Dropout`. Seeded reference values
  of both modules were pinned as regression values.
- `ragc` model-level checks (pre-consolidation suite) confirmed one kernelized hop
  equals the dense `(A x) / (A 1 + 1e-6)` computation at `atol=1e-5`, and that the
  stochastic shared embedding is identity in eval and train output rows are always
  original rows.
- The regression values were recorded from the extracted code; no official-code comparison is
  cited, so the match with the paper's operator is by structure only.

## Variants and options

- `order` sets the number of kernelized hops (all concatenated with the order-0 input).
- `dropout` default 0.0; the docstring notes the official reference uses 0.1.
- `StochasticSharedEmbedding` with `p=0` or in eval is a no-op; `ragc` can swap in
  `nn.Identity` when `use_sse` is false.
- Hops are non-negative weights over positive supports (the ReLU and softmax), so the
  implicit adjacency is symmetric and entrywise non-negative.

## Related components

`adaptive_node_embedding_adjacency` (same "adjacency from node embeddings" responsibility, but it materializes a dense row-softmax `[N, N]` adjacency in `O(N^2)`; this one never forms it, uses a gated-cosine support, and also applies the diffusion),
`diffusion_conv` (dense static-support diffusion), `sparse_connection_router`
(sparse learned adjacency over positions, in contrast with this implicit dense
kernelized one), `graph_utils` (static supports that a model may combine with this learned
one), `node_visibility` (another way `visifold`-style models cut node-attention cost;
random subsampling, not a graph operator).
