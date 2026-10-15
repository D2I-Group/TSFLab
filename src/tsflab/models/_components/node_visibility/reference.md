# node_visibility — reference

## Origin and granularity

Added with the `visifold` port as a new shared component; there was no
earlier duplicated code to consolidate. The functions are generic over what `L` indexes
(the docstring says nodes, patches, or any tokens), but `visifold` is the only consumer.
Kept model-local in `visifold`: the token construction, the `self.training` gating (the
pipeline runs only in training), the attention layers and output head, scattering
predictions for dropped nodes back to zero in the full `[B, N, pred_len]` output, and
restricting the loss to `keep_indices`.

## Invariants and equivalence evidence

- Contract check (at extraction): `kept == x[:, keep_indices]` with sorted, unique
  indices and the expected length; `mask_ratio=0` returns `x` itself with `arange(L)`;
  `mask_ratio=0.99` keeps one token; `mask_ratio` of `-0.1` or `1.0` raises `ValueError`;
  equal seeded generators give equal indices; `perm` is a per-row permutation and
  `unshuffle_tokens(shuffle_tokens(x))` returns `x`; grouping gives
  `[B * num_groups, subgraph_size, D]` with zero padding and `ungroup_subgraphs`
  restores `x`; `subgraph_size >= L` passes `x` through; `subgraph_size=0` raises
  `ValueError`; gradients through the full pipeline equal ones. Seeded reference
  values were pinned as a regression value.
- A `visifold` model-level check (pre-consolidation suite) ran the model in training
  mode through the full pipeline and checked output shape `(2, 3, 4)` and finiteness.
- The regression values were seeded from the extracted functions; there was no earlier
  implementation to compare against.

## Variants and options

- `mask_ratio=0` disables masking but keeps the index return value.
- `subgraph_size >= L` disables grouping.
- Pass a seeded `torch.Generator` for reproducible masks/shuffles (it must live on the
  same device as `x`).
- Masking is shared across the batch whereas shuffling is per sample; this asymmetry is
  part of the current behaviour.

## Related components

`graph_masked_attention` (dense/local node attention that benefits from smaller sets),
`self_attention_family` (the attention that runs on the grouped tokens),
`sparse_connection_router` (a learned, input-independent sparsification of node
interactions; this module is random, training-time subsampling and partitioning of
the token set instead).
