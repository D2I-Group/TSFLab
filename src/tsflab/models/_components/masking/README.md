---
name: "masking"
description: "Boolean attention-mask holders: TriangularCausalMask, ProbMask (ProbSparse query rows) and LocalMask (causal band); True marks a position filled with -inf. Use for causal or ProbSparse attention in Informer-lineage layers; not for padding masks, additive float masks, or graph masks."
---

# masking

## What it does

Three small classes that build boolean masks of the form used with
`scores.masked_fill_(mask, -inf)`: `True` means "may not attend".

- `TriangularCausalMask(batch_size, length, device)`: strict upper triangle
  (`j > i` masked), so position `i` sees `0..i`.
- `ProbMask(batch_size, num_heads, length, index, scores, device)`: the causal
  mask rows only for the selected (top-sparsity) queries `index`, as needed by
  ProbSparse attention.
- `LocalMask(batch_size, length, series_length, device)`: causal and limited to
  the previous `ceil(log2(length))` positions (band mask).

Each exposes the tensor through the `.mask` property.

## When to use

Use when a forecaster must not attend to future steps, such as the
decoder self-attention of encoder-decoder Transformers or ProbSparse attention in
the Informer-lineage layers of `self_attention_family`. Do not use for padding or
validity masks, for additive float masks (these are boolean fill masks), for
graph-structure masks (see `graph_masked_attention`), or in code that expects
`nn.Module` behaviour (device moves, state dict).

## Interface

`TriangularCausalMask(batch_size, length, device="cpu")`: `.mask` is bool
`[batch_size, 1, length, length]`.

`ProbMask(batch_size, num_heads, length, index, scores, device="cpu")`:
`length` is the number of query rows, `index` an integer tensor
`[batch_size, num_heads, n_selected]` of selected query rows (values below
`length`), `scores` `[batch_size, num_heads, n_selected, key_length]` (only its
shape is used). The mask for selected row `index` masks keys with position
greater than the row id. `.mask` is bool with the shape of `scores`.

`LocalMask(batch_size, length, series_length, device="cpu")`: `.mask` is bool
`[batch_size, 1, length, series_length]`; position `i` may attend to keys in
`[i - ceil(log2(length)), i]` (strict future and far past masked). Assumes
`series_length >= length` for a square-diagonal meaning.

General notes: plain Python classes (no `__call__`), not `nn.Module`s: no
parameters, buffers, state-dict keys, or training state. They do not validate
arguments; the mask is rebuilt on every construction (allocated on CPU, then
moved to `device`) and exposed read-only through `.mask`. The result is a bool
tensor and never requires grad. Import the three classes by name
(`TriangularCausalMask`, `ProbMask`, `LocalMask`; all are cataloged public symbols).
