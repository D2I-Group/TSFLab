---
name: "transformer_encdec"
description: "Post-norm Transformer encoder/decoder layers and stacks (conv-FFN) with optional Informer distilling ConvLayer and injected attention modules. Use for classic encoder-decoder forecasters (Transformer, Informer); not for patch encoders, pre-norm stacks, or decoders that return attention maps."
---

# transformer_encdec

## What it does

Standard post-norm Transformer blocks where the attention module is injected, so
the same stack serves full attention (`transformer`) and ProbSparse attention
with distilling (`informer`).

- `EncoderLayer`: `x = x + drop(attn(x,x,x)); x = LN1(x); y = conv2(drop(act(conv1(x))));
  out = LN2(x + y)`, where `conv1`/`conv2` are kernel-size-1 `Conv1d` (a
  position-wise FFN with width `d_ff = d_ff or 4*d_model`).
- `Encoder`: runs `attn_layers` in order, optionally interleaved with `conv_layers`
  (distilling), then an optional final `norm_layer`.
- `DecoderLayer`: self-attention, cross-attention to `cross`, FFN, with
  LayerNorms `norm1..norm3` after each residual.
- `Decoder`: layers, optional `norm_layer`, optional `projection`.
- `ConvLayer(c_in)`: Informer distilling: circular `Conv1d(k=3, padding=2)`,
  `BatchNorm1d`, `ELU`, `MaxPool1d(k=3, stride=2, padding=1)` over time.

## When to use

Use for classic encoder-decoder forecasters (vanilla Transformer, Informer) with
swappable attention and optional distilling that halves the encoder length
between layers. Do not use for patch encoders that want batch-first
`nn.TransformerEncoder` semantics (see `tst_transformer`), pre-norm stacks, or
decoders that must return attention maps (they are dropped).

## Interface

- `ConvLayer(c_in)`: parameters under `down_conv`, `norm`; length maps
  `L -> floor((L+1)/2) + 1` for the circular padding of 2 plus the stride-2 pool.
  Uses `BatchNorm1d`, so it has running statistics buffers.
- `EncoderLayer(attention, d_model, d_ff=None, dropout=0.1, activation="relu")`:
  `activation` must be `"relu"` or `"gelu"`, else `ValueError`. `forward(x, attn_mask=None, tau=None, delta=None) -> (x, attn)`; `attn` is the
  attention module's second return value. State-dict keys: `attention.*`,
  `conv1.*`, `conv2.*`, `norm1.*`, `norm2.*`. `dropout` in [0, 1) is not validated.
- `Encoder(attn_layers, conv_layers=None, norm_layer=None)`:
  `forward(x, attn_mask=None, tau=None, delta=None) -> (x, attns)`. With
  `conv_layers` the loop zips them with `attn_layers`, so `n-1` conv layers
  pair with the first `n-1` layers; the last attention layer is then called once
  more *without* `attn_mask`. In that
  distilling path `delta` is only given to the first layer; without `conv_layers`
  it is passed to every layer. `attns` has one entry per attention layer and holds
  whatever the attention returns (`None` unless it outputs weights). `conv_layers`
  must have exactly `len(attn_layers) - 1` entries, else `ValueError` (more would skip
  trailing attention layers, fewer would run the last one twice).
  State-dict keys: `attn_layers.{i}.*`, `conv_layers.{i}.*`, `norm.*`.
- `DecoderLayer(self_attention, cross_attention, d_model, d_ff=None, dropout=0.1,
  activation="relu")`: `forward(x, cross, x_mask=None, cross_mask=None, tau=None,
  delta=None) -> x`. State-dict keys: `self_attention.*`, `cross_attention.*`,
  `conv1.*`, `conv2.*`, `norm1.*`, `norm2.*`, `norm3.*`. `tau` reaches both attentions, `delta` only the cross
  attention; attention maps are discarded.
- `Decoder(layers, norm_layer=None, projection=None)`: same forward signature;
  applies the final norm then the projection. State-dict keys: `layers.{i}.*`,
  `norm.*`, `projection.*` (the last two only when supplied).

Attention modules must follow `attn(q, k, v, attn_mask=, tau=, delta=) ->
(out, attn)` over `[B, L, d_model]` (use `AttentionLayer`). Axis and dtype
follow the inputs; no module is stateful apart from `ConvLayer`'s batch-norm buffers.
Errors raised here: `ValueError` for an unsupported `activation` and a wrong `conv_layers` length.
