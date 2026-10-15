---
name: "flatten_forecast_head"
description: "Flatten the last two axes of [B, C, d, patches] and project to the horizon with one shared or one per-channel Linear, plus dropout (PatchTST head). Use after a patch or token encoder with a fixed patch count; not for autoregressive output, a run-time varying patch count, or channel mixing in the head."
---

# flatten_forecast_head

## What it does

For an input `x` of shape `[B, C, D, P]` the head computes, per channel,
`dropout(Linear(D * P -> H)(flatten(x[:, c])))` and returns `[B, C, H]`. With
`individual=False` one `Linear` and one dropout are shared across channels; with
`individual=True` channel `c` owns its own `Linear`, flatten, and dropout.
The last two axes are flattened in their existing order (`D` major, `P` minor).

## When to use

Use after a patch or token encoder whose output is arranged `[B, C, D, P]` and
whose forecast is a linear function of all `D * P` features per channel. Do not use
when the horizon should be produced autoregressively, when patch count changes at
run time (`nf` is fixed), or when channel mixing is required in the head.

## Interface

`FlattenForecastHead(individual: bool, n_vars: int, nf: int, target_window: int, head_dropout: float = 0.0)`

- `individual` (bool): per-channel parameters when true.
- `n_vars` (int >= 1): channel count. Only used when `individual=True` (loop bound); the shared head ignores it.
- `nf` (int >= 1): `D * P`, the flattened feature width; must equal the product of the last two axes of the input.
- `target_window` (int >= 1): horizon `H`.
- `head_dropout` (float in [0, 1)): dropout on the output.
- `forward(x)`: shared mode accepts any `[..., D, P]` (leading axes preserved) and returns `[..., H]`; individual mode requires rank 4 `[B, n_vars, D, P]` and returns the stack `[B, n_vars, H]`. A feature-width mismatch raises the `Linear` shape `RuntimeError`; no explicit validation.
- State-dict keys: shared `linear.weight`, `linear.bias`; individual `linears.{i}.weight`, `linears.{i}.bias`. Flatten and dropout have no parameters. Stateless apart from dropout.
