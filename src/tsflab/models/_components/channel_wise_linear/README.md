---
name: "channel_wise_linear"
description: "Linear map from input_length to output_length on [B, C, L], shared across channels or one nn.Linear per channel (LTSF-Linear). Use for channel-independent linear forecasts or heads with fixed lookback and horizon; not for cross-channel mixing, variable history length, or [B, L, C] input without a transpose."
---

# channel_wise_linear

## What it does

`ChannelWiseLinear(input_length, output_length, channels, individual)` maps the
history axis to the forecast axis. In shared mode one `nn.Linear(L, H)` is
applied to every channel; in individual mode channel `c` uses its own
`nn.Linear(L, H)`:

`y[b, c, :] = W_c x[b, c, :] + b_c` (with `W_c = W`, `b_c = b` for all `c` when shared).

Normalization, decomposition, and the transpose from `[B, L, C]` stay with the
caller.

## When to use

Use as the final temporal projection of a channel-independent linear or MLP
forecaster with a fixed history length and horizon. Do not use for variable
history length, when mixing across channels is needed (this layer never mixes
channels), or when the input is `[B, L, C]` without the caller transposing.

## Interface

`ChannelWiseLinear(input_length, output_length, channels, individual=False)`

- `input_length` (int >= 1), `output_length` (int >= 1), `channels` (int >= 1;
  only used for validation and, when `individual`, the number of layers).
  Constructor arguments are not validated; invalid values fail inside
  `nn.Linear`.
- Attributes `input_length`, `output_length`, `channels`, `individual`.
- Parameters and state-dict keys: shared mode `linear.weight` `[H, L]` and
  `linear.bias` `[H]`; individual mode `linears.{c}.weight` and
  `linears.{c}.bias` for `c` in `0..channels-1`. Weights use the default
  `nn.Linear` initialization (no `1/L` initialization is applied here;
  consumers like `cosa` overwrite `linear.weight` themselves).
- `forward(x)`: `x` is a floating tensor `[batch, channels, input_length]`.
  Raises `ValueError` if `x.ndim != 3` or `x.shape[1:]` differs from
  `(channels, input_length)`. Returns `[batch, channels, output_length]` in
  the dtype/device of the module.
- Stateless apart from the parameters. Individual mode loops over channels in
  Python, so it is slower for many channels.
