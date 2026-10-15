# sparse_connection_router — reference

## Origin and granularity

Added in commit `dd63af6c` (automated intake of SDMixer, SEMixer, LSINet, Dualformer) as a
new shared component taken from the `lsinet` port; no earlier copies were consolidated. The
`lsinet` test file ties it to the paper's shared connection matrix (Eqs. 3-6). It is cut as
the router alone. Kept model-local in `lsinet`: patching (Eq. 9 stride geometry), applying the
matrix to values (`connections @ values`, per-head value projection), the time-invariant
and time-updating MLPs, integration blocks, and any use of `probabilities`
for regularization (the model currently discards them).

## Invariants and equivalence evidence

- Contract check (at extraction): `[heads, N, N]` shapes and float32, that `_position_index` is not in the
  state dict, exactly `round(density * N^2)` ones per head in eval and in train mode
  (with Gumbel noise), values in `{0, 1}`, probabilities in `[0, 1]`, deterministic
  repeated eval calls, nonzero gradient to `memory.weight` and every `pair_scorer`
  parameter, the `ValueError` cases (`num_positions=0`, `heads=0`, `density` 0 and
  1.5), and `density=1.0` giving all ones; seeded reference values were pinned as
  regression values.
- `lsinet` model checks: the router is input-independent (two eval calls are
  identical), the learned interaction matrix is binary and respects the target
  density (values in `{0, 1}`, per-head ones count `max(1, round(density * N^2))`,
  probabilities in `[0, 1]`), and each block's multihead router is this class.
- These checks passed in the full suite run of 2026-10-03 before the test suite was
  consolidated.
- no fixture: there is no pre-refactor copy to compare against (the component was
  created directly for `lsinet`), so the only pinned reference is the seeded self-reference above.
  Not tested: `memory_dim != dim`, `gumbel_scale=0`, and tie behavior at the threshold.

## Variants and options

- The `bernoulli` and `gumbel-softmax` tags are retrieval aliases: there is no
  Bernoulli sampling; selection is a top-k threshold, perturbed in training by
  additive standard Gumbel noise (not a Gumbel-softmax relaxation).
- `heads` for independent matrices; `density` for target sparsity; `memory_dim` for
  embedding width; `gumbel_scale=0` for deterministic training-time selection.
- The sample-independent design is the point: for content-dependent sparsity compute scores
  from `x` instead (not provided).

## Related components

- `graph_masked_attention`: dense attention biased by a given adjacency; this
  router instead learns the (sparse, binary) adjacency and does not apply it to values.
- `node_visibility`: random token subsampling and subgraph grouping, not learned.
- `adaptive_node_embedding_adjacency`: dense learned adjacency from node embeddings
  (`softmax(relu(E1 E2^T))`); the same "learned adjacency" responsibility, differing in
  being dense, per-node-embedding, and with no density target or discrete output.
- `regularized_adaptive_graph_conv`: also learns an input-independent adjacency from
  node embeddings, but applies it as a linear-complexity graph convolution.
- `soft_tree`, `topk_expert_router`: other differentiable discrete routing (over
  tree leaves / experts, input-conditioned).
- `topk_expert_attention`: selects keys per query inside attention; this router selects connections between a learned memory and tokens.
- `weight_set_router`: input-independent softmax mixture of weight sets, no sparsity.
