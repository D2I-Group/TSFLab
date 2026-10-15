---
name: "differential_attention"
description: "Differential self-attention: softmax(Q1K1) minus learned lambda times softmax(Q2K2), applied to V, then per-head RMS renormalization; no projections or mask. Use for bidirectional mixing of variate or patch tokens when sharper attention is wanted; not for causal decoding, padding masks, or non-negative attention maps."
---

# differential_attention

## What it does

For one head, split `Q` and `K` along the last axis into halves `(Q1, Q2)`,
`(K1, K2)` of width `head_dim`:

`A = softmax(s * Q1 K1^T) - lambda * softmax(s * Q2 K2^T)`, `s = head_dim^-0.5`,
`lambda = exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init`

(per head, `lq*`, `lk*` learned vectors). The output is `A V` (with dropout on
`A`), then each head's output vector is divided by its RMS over the feature axis
(`sqrt(mean(x^2) + 1e-5)`), multiplied by learned `rms_scale` and by
`(1 - lambda_init)`.

## When to use

Use for bidirectional token mixing (for example variate or patch tokens) when a
sharper, noise-cancelled attention map is wanted, with an external projection
layer providing doubled `Q/K/V` widths. Do not use for causal decoding, when
the caller cannot supply `2*head_dim` per head, when a padding mask is needed,
or when attention maps must be non-negative (the difference can be negative).

## Interface

`DifferentialAttention(d_model, num_heads, lambda_init=0.8, attention_dropout=0.0)`:
`d_model % num_heads == 0` else `ValueError`; `head_dim = d_model // num_heads`.
Parameters (state-dict keys): `lambda_q1`, `lambda_k1`, `lambda_q2`, `lambda_k2`
(each `[num_heads, head_dim]`, init `N(0, 0.1^2)`), `rms_scale` (`[2*head_dim]`,
init 1, shared by all heads). `head_dim`, `num_heads`, `lambda_init`, `scale`
and `eps` are plain attributes. No other validation (`num_heads >= 1` is not
checked). `lambda` is unconstrained: it is not a convex combination and can be
negative or larger than one.

`forward(queries, keys, values)`: float tensors with `heads == num_heads`;
`queries` `[batch, seq, heads, 2*head_dim]`, `keys` and `values`
`[batch, key_seq, heads, 2*head_dim]` (`queries` and `keys` are each split in
half on the last axis; `values` keeps the full `2*head_dim`, which must match
`rms_scale`). `key_seq` may differ from `seq` (cross-attention works); `wdformer`
only uses self-attention. Returns `[batch, seq, heads, 2*head_dim]`. There are no
input/output projections and no shape validation (a wrong width fails inside
the chunk or einsum). No causal or padding mask: full bidirectional attention.
Dropout (active only in training mode) is applied to the differenced attention
map `A`, not to each softmax separately. Output dtype and device follow the
inputs.
