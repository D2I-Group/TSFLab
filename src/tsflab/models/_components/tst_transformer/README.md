---
name: "tst_transformer"
description: "Batch-first stack of torch nn.TransformerEncoderLayer blocks with a final BatchNorm or LayerNorm, used as the patch-token encoder of channel-independent forecasters. Use for [B*C, patches, d_model] token sequences; not for swappable attention cores, masks, or attention maps."
---

# tst_transformer

## What it does

`TSTEncoder(d_model, n_heads, ...)` encodes a sequence of tokens (for example
patch embeddings) with `n_layers` copies of `torch.nn.TransformerEncoderLayer`
(`batch_first=True`, `dim_feedforward=d_ff`, activation GELU by default,
`norm_first=pre_norm`) wrapped in `nn.TransformerEncoder`, followed by a final
norm that is either a `BatchNorm1d` over the feature axis (`_BatchFeatureNorm`)
or a `LayerNorm`. No positional information is added; callers add it before.

## When to use

Use as the encoder for `[B*C, patches, d_model]` token sequences in
channel-independent patch forecasters. Do not use when you need per-layer
BatchNorm, distinct key/value widths, attention maps, masks, or swappable
attention cores (use `transformer_encdec` with `self_attention_family`).

## Interface

The only public symbol is `TSTEncoder` (the module defines no `__all__` and the
catalog entry lists `TSTEncoder` as the public symbol; `_BatchFeatureNorm` and `_activation` are private).

`TSTEncoder(d_model, n_heads, *, n_layers=3, d_k=None, d_v=None, d_ff=256,
activation="gelu", norm="BatchNorm", attn_dropout=0.0, res_dropout=0.0,
ffn_dropout=0.0, proj_dropout=0.0, pre_norm=False, **_)`

- `d_model` divisible by `n_heads`, else `ValueError`; `n_layers` and `d_ff` are not
  validated here (torch errors apply).
- `d_k`/`d_v` must be `None` or `d_model // n_heads`, else `ValueError`.
- `activation`: `"relu"`, `"gelu"` (case-insensitive) or a callable, else `ValueError`.
- `norm`: any string containing `"batch"` or `"layer"` (case-insensitive), else
  `ValueError`; it selects only the final norm after the layer stack.
- The four dropout arguments collapse to one rate, `max(attn, res, ffn, proj)`,
  used for every dropout inside the torch layer.
- Extra keyword arguments are silently swallowed (`**_`), as are extra keywords to `forward`.

`forward(values, **_) -> Tensor` with `values` `[batch, tokens, d_model]`; output
has the same shape. State-dict keys: `layers.layers.<i>.*` (torch layer
parameters) and `layers.norm.*` (`layers.norm.norm.*`, including
`running_mean`/`running_var`/`num_batches_tracked` buffers, for the BatchNorm
variant). The BatchNorm variant is stateful in training (running statistics).
Float dtype/device follow the input.
