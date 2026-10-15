---
name: "hyper_state_scan"
description: "Scalar-state (d_state=1) selective scan returning the raw state trajectory, plus a depthwise 2-D conv that mixes that state over a (patch, variate) grid. Use for TimePro-style models that edit the SSM state before read-out; not for multi-state Mamba layers or long sequences."
---

# hyper_state_scan

## What it does

`diagonal_selective_scan(u, delta, a, b)` evaluates, per batch and channel, the
scalar recurrence `h_t = exp(delta_t * a) * h_{t-1} + delta_t * b_t * u_t` with
`h_0 = 0` and returns every `h_t`. Unlike the `mamba` scan it does not apply the
`C` read-out or the `D` skip, so the caller can edit the raw state first.
`GridStateMixer` applies a depthwise `Conv2d` over a channel-first grid to mix each
channel's state with its spatial neighbours.

## When to use

Use when a state-space model should carry information along a token axis (in
`timepro`, the variate axis) and the per-token state must be mixed with its
neighbours on a (time-patch, variate) grid before the read-out. Do not use for
`d_state > 1`, when a read-out and skip inside the scan are wanted (use
`mamba`), for very long token axes (sequential Python loop), or when true
deformable sampling is required (the mixer is a fixed local conv).

## Interface

`diagonal_selective_scan(u, delta, a, b) -> Tensor`: `u`, `delta`, `b` all
`[batch, channels, length]` with identical shapes and `length >= 1` (a zero-length axis raises `ValueError`); `a` is `[channels]` (already
negative, i.e. `-exp(a_log)`). Returns `[batch, channels, length]`; `h[..., t]` is
the state after position `t`. Raises `ValueError` if `u`, `delta`, `b` shapes differ, the length axis is empty, or
`a` is not 1-D with length `channels`. Pure function; `u`, `delta`, `a`, `b` must share dtype and device; sequential Python loop over
`length`, differentiable, no parameters.

`GridStateMixer(channels, kernel_size=3)`: `channels >= 1`, `kernel_size` positive
odd (else `ValueError`). Holds `self.conv`, a depthwise `Conv2d` with `groups=channels`,
`padding=kernel_size//2`, so the spatial size is preserved; state-dict keys
`conv.weight` `[channels, 1, k, k]` and `conv.bias` `[channels]`. `forward(grid)`
requires `ndim == 4` `[batch, channels, height, width]` (else `ValueError`);
height and width are caller-defined. Zero padding at the grid border; channels
never mix with one another.
