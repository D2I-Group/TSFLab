---
name: "gated_dilated_conv"
description: "Causal left-padding and the WaveNet gated activation tanh(filter(x)) * sigmoid(gate(x)) over caller-owned dilated conv modules, on [B, C, T] or [B, C, N, T]. Use for stacks of causal dilated temporal convolutions (WaveNet, Graph WaveNet); not for non-causal context, mismatched filter and gate, or GLU-style gating."
---

# gated_dilated_conv

## What it does

`causal_pad(x, dilation, kernel_size)` zero-pads `dilation * (kernel_size - 1)`
steps on the left of the last (time) axis only. `gated_dilated_conv(x, filter_conv,
gate_conv)` pads `x` this way, using `filter_conv`'s last-axis `kernel_size` and
`dilation`, and returns `tanh(filter_conv(p)) * sigmoid(gate_conv(p))` with
`p = causal_pad(x)`. If the convs have no padding, the output has the same time
length as `x` and position `t` depends only on inputs up to `t`.

## When to use

Use for WaveNet-style temporal gating in a stack of dilated causal convs, on
`[B, C, T]` or `[B, C, N, T]` layouts. Do not use when the module should own its
convolutions, when the filter and gate differ in kernel or dilation, when
non-causal context is wanted, or for other gating forms (GLU, see `gated_fusion`).

## Interface

`causal_pad(x, dilation, kernel_size) -> Tensor`: `x` rank 3
`[batch, channels, time]` or rank 4 `[batch, channels, nodes, time]`; any other
rank raises `ValueError`. `dilation` and `kernel_size` are ints >= 1. Pads the last
axis on the left only.

`gated_dilated_conv(x, filter_conv, gate_conv) -> Tensor`: `filter_conv` and
`gate_conv` are caller-owned `nn.Conv1d` (rank-3 `x`) or `nn.Conv2d` (rank-4 `x`,
kernel such as `(1, k)`, dilation `(1, d)`) modules. Only `filter_conv.kernel_size[-1]`
and `filter_conv.dilation[-1]` are read for padding; `gate_conv` must use the same
last-axis kernel and dilation and the same output channels (not checked). Their own
`padding` should be 0 along time, otherwise the result is not causal. Output
channels are those of the convs; the time length equals the input's. No parameters,
buffers or state of its own; no dtype handling beyond the convs'.
