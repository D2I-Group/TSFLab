---
name: "last_value_center"
description: "Stateless NLinear-style pair: subtract the detached last time step from a [B, L, C] history and add it back to the head output. Use for level shifts and drifting means in front of linear or recurrent heads; not for scale drift (use revin) or heads that change the channel layout."
---

# last_value_center

## What it does

Two stateless functions that remove and restore the most recent level of a
history window. With `level = x[:, -1:, :].detach()`:

- `center_on_last_value(x) -> (x - level, level)`;
- `restore_last_value(head_output, level) -> head_output + level`.

No scaling is applied. The detach means no gradient reaches the model through
the additive level term (the centered path still gets gradient through `x`).

## When to use

Use when the series level drifts or jumps (non-stationary means, a level shift
between training and evaluation windows) and a cheap correction is enough: the
head forecasts the change from the last observation. Do not use when the scale
also drifts (use `revin`), when the last step is noisy enough to bias the whole
forecast, when the output channel layout differs from the input, or when
gradient through the level is required.

## Interface

`center_on_last_value(x)`

- `x`: floating tensor `[batch, length, channels]` (indexed as
  `x[:, -1:, :]`; ranks other than 3 are not validated). `length >= 1`.
- Returns `(centered, level)`: `centered` has the shape of `x`; `level` is
  `[batch, 1, channels]`, detached (`requires_grad=False`).

`restore_last_value(head_output, level)`

- `head_output`: `[batch, horizon, channels]`; `level` broadcastable to it
  (a Python float such as `0.0` also works). Returns the sum with the
  broadcast shape. No validation, no parameters, no state-dict keys, no errors
  raised by the component.
