---
name: "patchtst"
description: "Channel-independent PatchTST backbone: RevIN, patching, linear patch embedding, position table, TSTEncoder, flatten head; direct multi-horizon point output. Use for a reusable backbone on weakly correlated channels (e.g. under a quantile head); not for channel mixing, covariates, or matching the patchtst model."
---

# patchtst

## What it does

`PatchTSTBackbone` is the whole PatchTST forward pass as a reusable module.
For input `x [B, L, C]`:

1. `RevIN` normalization (`norm`), then `[B, C, L]`.
2. Optional end padding: replicate-pad `stride` steps when `padding_patch="end"`.
3. `unfold(-1, patch_len, stride)` gives patches `[B, C, N, patch_len]` with
   `N = 1 + (L' - patch_len) // stride`, `L' = L + stride` if padded.
4. Linear `patch_len -> d_model`, add the `[N, d_model]` position table, input dropout.
5. Fold channels into the batch (`[B*C, N, d_model]`), run `TSTEncoder`
   (all channels share weights: channel independence).
6. Back to `[B, C, d_model, N]`, `FlattenForecastHead` (flatten `d_model * N` then
   linear to `target_window`, shared or per-channel via `individual`),
   transpose to `[B, H, C]`, RevIN `denorm`.

## When to use

Use as the point-forecast backbone when channels are best modelled
independently with shared weights (weakly correlated channels), for example
under a quantile head. Do not use when cross-channel mixing, marks or exogenous
inputs matter, for masked pretraining, or to reproduce the model-local `patchtst`
numerics (the `patchtst` model does not run this component).

## Interface

`PatchTSTBackbone(c_in, context_window, target_window, patch_len, stride,
padding_patch, n_layers, d_model, n_heads, d_k, d_v, d_ff, activation, norm,
attn_dropout, res_dropout, ffn_dropout, proj_dropout, head_dropout, pre_norm,
pe, learn_pe, head_type, individual, revin, affine, subtract_last)`: every
argument is required (no defaults). Meaning: `c_in` channels; `context_window`
history length `L` (exact); `target_window` horizon `H`; `patch_len >= 1`,
`stride >= 1`, `patch_len <= context_window`; `padding_patch` is `None` or
`"end"`; encoder arguments are forwarded to `TSTEncoder` (`d_k`/`d_v` must be
`None` or `d_model // n_heads`); `pe` / `learn_pe` go to `positional_encoding`;
`head_type` must be `"flatten"`; `individual` selects per-channel heads;
`revin`, `affine`, `subtract_last` configure `RevIN` (`revin=False` disables it).

`forward(values, *_)`: `values` float `[B, context_window, c_in]`; extra
positional arguments (for example marks) are ignored. Returns `[B, target_window,
c_in]`. Raises `ValueError` for bad shape/length, `patch_len`/`stride`, unsupported
`padding_patch`, or `head_type != "flatten"` (plus errors from the sub-components).
State-dict prefixes: `normalizer.*` (RevIN affine), `patch_projection.*`,
`position` (the table, an `nn.Parameter` with `requires_grad=learn_pe`),
`encoder.layers.layers.*` and `encoder.layers.norm.*` (an `nn.TransformerEncoder`
wrapped by `TSTEncoder`, with `norm="BatchNorm"` adding running-statistics buffers),
`head.*`. Dropout: `TSTEncoder` collapses `attn_dropout`, `res_dropout`,
`ffn_dropout` and `proj_dropout` into one rate, their maximum
(so they are not independently tunable); `res_dropout` also drops the embedded
tokens, `head_dropout` the flattened features. The module is stateful during a
forward pass through the RevIN statistics cache (detached, so no gradient flows
through the instance mean/std) and the encoder's BatchNorm running statistics;
do not share one instance across concurrent forwards. `denorm` is only valid
after a `norm` in the same forward, which the module guarantees.
