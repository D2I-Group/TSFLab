---
name: "self_attention_family"
description: "Time-Series-Library-style attention cores (full softmax, ProbSparse, flow, tiled flash-style, Reformer LSH) plus AttentionLayer with Q/K/V/output projections. Use for Informer/Transformer encoder-decoders that swap the attention core; not for patch encoders, score biases, or deterministic ProbSparse."
---

# self_attention_family

## What it does

A bundle of interchangeable attention "cores" sharing one call signature
`core(queries, keys, values, attn_mask, tau=None, delta=None) -> (output, attn_or_None)`
and one wrapper, `AttentionLayer`, that projects a `d_model` sequence into
heads, calls a core, and projects back.

- `FullAttention`: `softmax(scale * Q K^T) V` per head with `scale = scale or
  1/sqrt(d_head)`, optional causal masking, attention dropout.
- `ProbAttention`: Informer ProbSparse attention. Samples `u_part = factor *
  ceil(ln len_k)` keys per query, scores queries by `max - mean` of the sampled
  logits, keeps the top `u = factor * ceil(ln len_q)` queries, runs exact
  attention only for those, and fills the remaining query rows with the mean of
  `V` (`mask_flag=False`) or the running cumulative sum of `V` (`mask_flag=True`).
- `FlowAttention`: sigmoid-kernel linear attention with row/column flow
  normalizers; no softmax map is produced and `attn_mask` is ignored.
- `FlashAttention`: pure-PyTorch block-wise online-softmax attention (an
  algorithmic tiling reference, not a fused kernel).
- `ReformerLayer`: thin wrapper over `reformer_pytorch.LSHSelfAttention`; it is a
  full self-attention layer (own projections), not a core, and no model imports it
  (`reformer` has its own local LSH attention).
- `AttentionLayer(attention, d_model, n_heads, d_keys=None, d_values=None)`.

## When to use

Use for encoder/decoder Transformers in the Informer/Transformer line where the
attention core must be swappable behind `AttentionLayer`: exact attention for
short token sequences, ProbSparse or flow attention when quadratic cost over long
sequences matters. Do not use `ProbAttention` for deterministic evaluation or
short sequences (it samples even when it falls back to all queries); do not use
`FlashAttention` as a performance primitive (it is a reference). For patch-token
encoders use `tst_transformer`; for noise-cancelling or routed variants see
`differential_attention` and `topk_expert_attention`.

## Interface

Cores `FullAttention`, `ProbAttention`, `FlashAttention`: `(mask_flag=True,
factor=5, scale=None, attention_dropout=0.1, output_attention=False)`; `factor`
is used only by `ProbAttention`, and `FlashAttention` ignores all five.
`FlowAttention(attention_dropout=0.1)`. `ReformerLayer(attention, d_model,
n_heads, d_keys=None, d_values=None, causal=False, bucket_size=4, n_hashes=4)`
needs the optional `reformer_pytorch` package. All cores are stateless and
parameter-free; per-core argument quirks are in reference.md.

`AttentionLayer(attention, d_model, n_heads, d_keys=None, d_values=None)`:
`d_keys`/`d_values` default to `d_model // n_heads`. State-dict keys:
`query_projection`, `key_projection`, `value_projection`, `out_projection`, plus
`inner_attention.*` of the core. `forward(queries, keys, values, attn_mask,
tau=None, delta=None)` takes batch-first `[B, L, d_model]` and returns
`(out [B, L_q, d_model], attn)`.

`attn_mask` is an object with a `.mask` bool tensor (see `masking`);
`FullAttention` with `mask_flag=True` and no mask builds a causal one, while
`FlashAttention` takes a `[batch, len_k]` 0/1 key-padding tensor instead.
`tau`/`delta` are accepted and ignored. `ProbAttention` sampling is stochastic
even in eval mode. Mask, attention-map and layout details: reference.md.
