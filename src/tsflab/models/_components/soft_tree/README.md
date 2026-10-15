---
name: "soft_tree"
description: "Differentiable depth-d binary trees: sigmoid split probabilities along each path interpolate learned leaf vectors; plain and oblivious forms. Use for tree and boosting baselines on flattened lag features; not for rank-3 sequences, large depth, or faithful greedy CART/GBDT."
---

# soft_tree

## What it does

A depth-`D` tree has `2^D - 1` internal nodes (heap order, root 0, children
`2n+1`/`2n+2`) and `2^D` leaves. For `x: [B, F]`, node `n` sends `x` right with
probability `r_n = sigmoid((x @ (W_n * mask_n) - theta_n) / T)`. A leaf's
probability is the product over its path of `r_n` (right branch) or `1 - r_n`
(left); the output is `leaf_probs @ leaf_value` (`leaf_value: [2^D, output_dim]`).
Leaf probabilities sum to 1 per row. `SoftObliviousTree` uses one split per depth
level (`split_weight: [D, F]`, `threshold: [D]`) shared by every node of that level.

## When to use

Use for differentiable tree or tree-ensemble baselines over flat `[B, F]`
features (for example a flattened lag window). Do not use for rank-3 sequences
without flattening, for large depth (leaves grow as `2^D`), or when a faithful
tree algorithm (greedy splits, exact boosting) is needed.

## Interface

`SoftDecisionTree(input_dim, output_dim, depth=3, temperature=1.0, *, split_mask=None,
fixed_split_weight=None, fixed_threshold=None)`: all dims and `depth` >= 1,
`temperature > 0` (else `ValueError`). `split_mask` `[2^D-1, F]` (default all ones)
must keep at least one feature per node. `fixed_split_weight` `[2^D-1, F]` and
`fixed_threshold` `[2^D-1]` must be given together; then they are buffers (frozen)
and only `leaf_value` trains; otherwise `split_weight` (normal, std 1/sqrt(F)) and
`threshold` (zeros) are parameters. State-dict keys: `split_weight`, `threshold`,
`leaf_value` (init normal, std 0.02), buffers `split_mask`, `route_nodes`,
`route_right` (the last two are structural and are saved too; with fixed splits
`split_weight`/`threshold` are buffers rather than parameters).
`SoftObliviousTree(input_dim, output_dim, depth=3, temperature=1.0)`: keys
`split_weight` `[D, F]`, `threshold` `[D]`, `leaf_value`, buffer `route_right`.
Both: `leaf_probabilities(x)` and `forward(x)` need `x` of shape `[batch, input_dim]`
(otherwise `ValueError`); leaves are ordered by path bits, most significant bit
first (leaf 0 is all-left). `binary_routes(depth)` returns heap node indices and
right-branch flags for every leaf (`ValueError` for `depth < 1`). Mask is applied
as `split_weight * split_mask` at every forward. No state across calls. Cost grows as `2^D`; keep depth small. Buffers and parameters are float32, so a float64 input needs `.double()` on the module (otherwise the matmul raises).
