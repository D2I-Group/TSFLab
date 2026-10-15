---
name: "gated_fusion"
description: "Convex blend of two equal-shaped tensors with a learned sigmoid gate: g = sigmoid(Wa a + Wb b), out = g * a + (1 - g) * b. Use for fusing two same-shaped branch embeddings per feature; not for differently shaped branches, additive or concatenative fusion, or a gate shared across features."
---

# gated_fusion

## What it does

`GatedFusion(dim)` computes an elementwise gate from both inputs and returns a
convex combination:

`gate = sigmoid(Linear_a(a) + Linear_b(b))`, `out = gate * a + (1 - gate) * b`.

Every output element lies between the corresponding elements of `a` and `b`.
Both linears are `dim -> dim` with bias, and the gate is per position and per feature.

## When to use

Use when two branches produce same-shaped embeddings and a data-dependent convex
blend per feature is wanted. Do not use when the branches have different shapes
(project them first), when the fusion should be additive or concatenative, or when
the gate should be shared across features.

## Interface

`GatedFusion(dim: int)`

- `dim` (int >= 1): width of the last axis; `dim < 1` raises `ValueError`.
- `forward(a, b) -> Tensor`: `a` and `b` must have exactly equal shapes (any rank, last axis `dim`), else `ValueError`. Output has the same shape, dtype, and device. The usual call in `gateformer` is on `[B, n_vars, d_model]`.
- State-dict keys: `gate_a.weight`, `gate_a.bias`, `gate_b.weight`, `gate_b.bias` (`[dim, dim]` or `[dim]`). Parameters only, no buffers, no dropout, no randomness.
- The two inputs are not symmetric in parameters (`Wa` acts on `a`, `Wb` on `b`) but the blend is: the gate is the weight on `a`.
