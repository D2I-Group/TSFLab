---
name: "frwkv_linear_attention"
description: "FRWKV's RWKV-7-style delta-rule linear attention (decay, key removal, value-key write, bonus, gated output) and its residual frequency branch. Use for FRWKV-family frequency encoders or an order-dependent mixer over few tokens; not for unordered tokens, faithful RWKV-7 time mixing, or long token sequences."
---

# frwkv_linear_attention

## What it does

`FRWKVLinearAttention` is the token mixer of the FRWKV frequency encoders: an
RWKV-7-style per-head matrix state updated once per token by
`S_t = (diag(w_t) - k~_t (a_t * k~_t)^T) S_{t-1} + v_t k^_t^T` and read out as
`y_t = S_t r_t`, then a bonus term `0.2 (r . k^) b * v`, a LayerNorm, a sigmoid
output gate and a two-layer GELU output projection. Cost is linear in the number
of tokens. `FRWKVSpectralBranch` wraps a stack of these layers into the residual
branch `x + Linear_out(Encoder(Linear_in(x)))` that FRWKV applies separately to the
real and the imaginary spectrum.

## When to use

Use to reproduce FRWKV-family frequency encoders, or as an order-dependent,
linear-cost token mixer whose recurrence runs over a modest number of tokens (the
scan is a Python loop over tokens). Do not use where tokens are unordered and
permutation equivariance is expected, where a faithful RWKV-7 time mixer is
needed, or for long token sequences where a sequential loop is too slow.

## Interface

- `FRWKVLinearAttention(d_model, n_heads, dropout=0.0)`: `d_model` must be a
  positive multiple of `n_heads` and `dropout` in `[0, 1)`, else `ValueError`.
  `forward(queries, keys=None, values=None, attn_mask=None, tau=None, delta=None)`
  mixes `queries` `[batch, tokens, d_model]` with itself; keys, values and mask
  are accepted for the attention-layer signature and ignored. Returns
  `(output, None)`; `dropout` applies once to the output projection. A wrong last
  width raises `ValueError`. `decay(x)` returns the per-channel decay used by the
  scan. Parameters: `mu_{r,k,v,g,a,w}`, `decay_lora`, `strength_lora`,
  `gate_lora`, `receptance`, `key`, `value`, `output`, `removal_scale`,
  `strength_mix`, `bonus`, `norm`.
- `FRWKVSpectralBranch(in_features, d_model, d_ff, n_heads, e_layers, dropout=0.0, activation="gelu")`:
  `Linear(in_features, d_model)`, `e_layers` `transformer_encdec.EncoderLayer`
  blocks (conv feed-forward of width `d_ff`, layer dropout `dropout`) around
  `FRWKVLinearAttention(dropout=dropout / 2)`, a final LayerNorm, and
  `Linear(d_model, in_features)`, added to the input. `forward(x)` keeps the
  shape `[batch, tokens, in_features]`; tokens are mixed in their given order.
- `frwkv_state_scan(r, removal_key, replacement_key, v, decay, strength)`: the
  bare recursion on `[batch, tokens, heads, head_size]` tensors (removal key
  already unit-normalized), zero initial state, returns `y` of the same shape.
- No buffers; no state between calls.
