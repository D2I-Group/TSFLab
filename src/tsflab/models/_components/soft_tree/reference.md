# soft_tree — reference

## Origin and granularity

Introduced as the shared primitive of the tree-ensemble baselines in the rewrite
commit `dc15d37d` ("replace tree and boosting adapters"), when the local
`soft_tree.py` held the algebra and every consumer kept only its ensemble
logic. No paper for the soft-tree algebra is recorded in history. Model-local: ensemble
construction (averaging for forest/extra-trees, stagewise residual stacking for
the boosting models, CatBoost's forecast-context linear maps), random seeds and
feature subsets, RevIN, and any training interpretation. These are differentiable
approximations, not the original greedy CART/GBDT algorithms.

## Invariants and equivalence evidence

- Contract checks (at extraction): `binary_routes` pins the depth-2
  node and right-flag tables, dtypes, and the `depth=0` error; for both classes,
  `[4, 8]` leaf probabilities that are non-negative and sum to 1, `[4, 3]` output,
  input and parameter gradients, and the `ValueError`s for wrong input rank or width
  and for invalid dimensions or temperature; the exact state-dict keys of both
  classes, that the oblivious `route_right` equals `binary_routes(2)[1]`, and seeded
  values pinned as a regression value; that masked-out features do not
  affect the output, the mask and fixed-split `ValueError`s, and that fixed splits leave
  only `leaf_value` trainable while `split_weight`/`threshold` stay in the state dict.
- Tree-baseline model checks: the depth-1 closed form (exact sigmoid leaf
  interpolation) with temperature 2 and leaf probabilities summing to one; the
  oblivious tree sharing one decision per depth; and runs of the consumers
  (distinct compositions, complete runtime contract). These checks passed in the
  full suite run of 2026-10-03 before the test suite was consolidated.

## Variants and options

`split_mask` for fixed per-node feature subsets (random forest), fixed split
weights and thresholds for randomized frozen geometry (extra trees),
`SoftObliviousTree` for symmetric trees (CatBoost style), `temperature` for
routing sharpness (small is closer to hard splits). No hard routing, no
multi-output-per-leaf variants beyond a vector `leaf_value`.

## Related components

`topk_expert_router` (also routes by learned scores, but picks a sparse top-k set of
experts per token; `soft_tree` is dense, every leaf gets a nonzero probability, and
the routes are a fixed binary tree), `sparse_connection_router` (learned sparse
connectivity over positions, not leaf interpolation), `revin` (all seven consumers
normalize with it).
