---
name: "weight_set_router"
description: "Low-rank weight sharing (DiPE-Linear): temperature-softmax routing over M weight sets and the per-channel convex mix of those sets. Use for many channels that should share a few parameter sets instead of fully shared or per-channel weights; not for input-dependent or top-k routing."
---

# weight_set_router

## What it does

`WeightSetRouter(num_sets, channels)` owns a learnable matrix `R` of shape
`[num_sets, channels]` (standard normal initialization) and returns
`R' = softmax(R / tau, dim=0)` for a caller-supplied temperature `tau`, so every
channel receives a convex combination of `num_sets` weight sets.
`mix_weight_sets(weights, routing)` forms `G_c = sum_m R'[m, c] * G_m`, turning
`[num_sets, *shape]` weights into `[channels, *shape]`. With `num_sets` much
smaller than `channels` this interpolates between a fully shared and a fully
per-channel parameterisation.

## When to use

Use when many channels should neither share one set of weights nor each get
their own: a small number of parameter sets is mixed per channel with a learned,
input-independent assignment. Do not use for input-dependent expert selection or
sparse top-k routing (see `topk_expert_router`), or when the channel count varies.

## Interface

`WeightSetRouter(num_sets, channels)`: sizes must be positive. Parameter
`logits` `[num_sets, channels]`. `forward(temperature=1.0)` requires
`temperature > 0` and returns `[num_sets, channels]` whose columns sum to 1.

`mix_weight_sets(weights, routing)`: `weights` is `[num_sets, ...]`, `routing`
is `[num_sets, channels]`; a non-2-D `routing` or mismatched set count raises
`ValueError`. Returns `[channels, ...]` (dtype by promotion of the two inputs) and is
differentiable in both arguments. Routing need not be softmax-normalized; any
`[num_sets, channels]` matrix is accepted.

Both symbols are module-level (`__all__` lists them). `logits` is float32 on the
module's device; the module is stateless apart from this parameter. Input-independent:
the routing does not depend on any data tensor.
