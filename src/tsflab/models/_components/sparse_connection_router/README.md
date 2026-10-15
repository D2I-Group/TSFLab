---
name: "sparse_connection_router"
description: "Input-independent learned binary connection matrix per head over N positions: pair-MLP scores, top-k density, straight-through gradient, Gumbel noise in training (LSINet). Use for a lightweight shared sparse topology over patches; not for input-dependent routing, variable or large N."
---

# sparse_connection_router

## What it does

Learns, per head, a sparse 0/1 `N x N` matrix saying which of `N` positions (e.g. time
patches) exchange information. The matrix depends only on learned parameters, never on
the input batch, so one matrix is shared across all samples and channels.

Per call: a learnable table (`N` positions x `memory_dim * heads`) is split per head and
passed through a 3-layer linear encoder; every ordered pair `(i, j)` is scored by an MLP
over `concat(m_i, m_j)` giving `scores [heads, N, N]`; `probabilities = sigmoid(scores)`.
The hard matrix keeps the top `k = max(1, round(density * N * N))` entries per head of
`scores` (plus `gumbel_scale * Gumbel` noise in training), returned with the
straight-through form `hard + (soft - soft.detach())` where `soft = sigmoid(gated_scores)`.

## When to use

Use for a fixed number of positions (time patches) where one learned sparse
topology, shared across samples and channels, should replace attention in a
lightweight model. Do not use for variable `N`, for input-dependent routing, for
large `N` (the pairwise tensor is `N^2`), or when gradients must reach `x` (the
router never sees it).

## Interface

`SharedSparseConnectionRouter(num_positions, dim, heads=1, density=0.15, memory_dim=None, gumbel_scale=1.0)`

- `num_positions` (int >= 1) and `heads` (int >= 1); `density` in `(0, 1]`; violations raise
  `ValueError`. `memory_dim` defaults to `dim` (`dim` is used only for this default).
  `gumbel_scale` (>= 0): noise multiplier in training; `0` disables noise.
- `forward() -> (connections, probabilities)`: no arguments, both `[heads, N, N]` float.
  `connections` is numerically 0/1 with value gradients flowing through the sigmoid;
  the number of ones per head is `round(density * N * N)` (at least 1), except that ties at
  the threshold would add extra ones. `probabilities` is the sigmoid of the noise-free scores.
- Parameters/state-dict keys: `memory.weight [N, memory_dim * heads]`,
  `memory_encoder.{0,1,2}.{weight,bias}` (Linear `memory_dim -> memory_dim//2 -> memory_dim//4 ->
  memory_dim`, minimum width 1, with no nonlinearities between them), and `pair_scorer.{0,2}.{weight,bias}`
  (Linear `2*memory_dim -> memory_dim`, ReLU, Linear `memory_dim -> 1`). The non-persistent buffer
  `_position_index` is not in the state dict.
- Memory cost is `O(heads * N^2 * memory_dim)` for the pair tensor. Randomness: the global torch RNG,
  only when `self.training` and `gumbel_scale > 0`; in eval `forward` is deterministic.
