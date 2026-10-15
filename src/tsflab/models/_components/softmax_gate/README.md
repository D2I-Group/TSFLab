---
name: "softmax_gate"
description: "Softmax feature gate: out = x * softmax(Linear(x), dim=-1) over the last axis, no residual or normalization (PatchTSMixer gated attention). Use for filtering features after an MLP mixer; not for blending two inputs, gates that must not sum to one, or a non-last axis."
---

# softmax_gate

## What it does

`SoftmaxGate(dim)` computes `w = softmax(Linear(x), dim=-1)` and returns `x * w`
(elementwise). Because `w` sums to one over the feature axis, it acts as a
feature-importance filter: dominant features are kept, the rest are damped. The
paper calls this "gated attention" although it has no query/key interaction.

## When to use

Use to filter the features of a tensor by a learned softmax distribution along
its last axis, typically after an MLP mixer and before a residual add. Do not use
for a two-input blend (`gated_fusion`), for a gate that must not sum to one
across features, or when the gated axis is not last (permute first).

## Interface

`SoftmaxGate(dim: int)`

- `dim` (int >= 1): width of the last axis; `dim < 1` raises `ValueError`.
- `forward(x [..., dim]) -> Tensor`: same shape and dtype; any number of leading axes. The input must share dtype and device with the module parameters (float32 by default). A last axis other than `dim` raises `ValueError`.
- State-dict keys: `score.weight` `[dim, dim]`, `score.bias` `[dim]`. No dropout; stateless apart from parameters.
