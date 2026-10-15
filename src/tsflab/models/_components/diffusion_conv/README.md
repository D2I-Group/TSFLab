---
name: "diffusion_conv"
description: "Graph WaveNet diffusion convolution on [B, C, N, T]: concatenate x and its 1..order hops under each dense support, then a 1x1 conv and dropout. Use for spatiotemporal data with fixed or adaptive node graphs; not for sparse or per-sample dynamic supports, layouts without a time axis, or a varying support count."
---

# diffusion_conv

## What it does

Mixes information across graph nodes at every time step. For supports
`S_1..S_m` (`m = support_len`) and hop count `K = order`, with `x` shaped
`[B, C, N, T]`:

```
out = Dropout( Conv1x1( concat_channels[ x, S_1 x, S_1^2 x, .., S_1^K x, S_2 x, .., S_m^K x ] ) )
```

where `S x` means `einsum("ncvl,vw->ncwl", x, S)` (the support is applied on the
node axis as `x @ S`, so row `v` of `S` is the source node). The concatenation has
`(K * m + 1) * c_in` channels.

## When to use

Use for `[B, C, N, T]` spatiotemporal features with a fixed list of dense `[N, N]`
supports (forward/reverse transition plus an adaptive one). Do not use for
`[B, N, C]` layouts without a time axis (reshape first), for sparse supports
(they are dense einsums, `O(N^2)` per hop), for dynamic per-sample supports (the
einsum `vw` is shared across the batch), or when the support count may vary at runtime.

## Interface

`DiffusionConv2d(c_in, c_out, dropout, support_len=3, order=2)`

- `c_in`, `c_out` (int >= 1): input and output channels.
- `dropout` (float in [0, 1)): applied with `F.dropout` to the projected output only
  while `self.training`; no range validation.
- `support_len` (int >= 1): exact number of supports `forward` must receive.
- `order` (int >= 1): hops per support. Raises `ValueError` otherwise.
- `forward(x, supports)`: `x` is `[B, c_in, N, T]`; `supports` is a list of
  `[N, N]` tensors; each is moved to `x.device` (not dtype). Raises `ValueError` when
  `len(supports) != support_len`. Output `[B, c_out, N, T]`.
- Parameters (state-dict keys): `mlp.mlp.weight` `[c_out, (order*support_len+1)*c_in, 1, 1]`
  and `mlp.mlp.bias` `[c_out]`. No buffers, no persistent state.

Helper modules also exported: `NeighborhoodConv2d()` with
`forward(x, support)` computing one `einsum("ncvl,vw->ncwl")` hop (no parameters), and
`PointwiseProjection(in_channels, out_channels)`, a `Conv2d` with kernel `(1, 1)`
stored as attribute `mlp` (hence the doubled `mlp.mlp` prefix).
