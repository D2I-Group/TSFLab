# differential_attention — reference

## Origin and granularity

Added with WDformer in the automated model intake (`d73cf9d0`); `wdformer` is
the only consumer. It is cut as the core operator only: `wdformer` keeps the
`DifferentialSelfAttentionLayer` that projects `d_model` to the doubled `Q/K/V`
widths (`2*d_model`), splits heads, and projects back with `out_proj`, plus its
own `RMSNorm`, SwiGLU feed-forward and encoder layer. `lambda_init` is a
constructor constant here; the depth schedule is model-local: `wdformer`
computes one value per encoder layer (`0.7 - 0.5 * exp(-0.3 * layer)`) and passes
it in.

## Invariants and equivalence evidence

- A contract check verified state-dict keys and parameter shapes, output shape
  and dtype, finite gradients for `queries`, `keys`, `values` and all
  parameters, and the `ValueError` for `d_model` not divisible by `num_heads`;
  the seeded output was pinned as a regression value. A further check showed
  that changing the last value token changes the first output position (no
  causal mask).
- A `wdformer` structure check verified forward/backward over all parameters
  finite (including `lambda_*` and `rms_scale`), a strict state-dict round trip,
  and `attention.attention._lambda()` finite. These checks passed in the full
  suite run of 2026-10-03 before the test suite was consolidated.
- No reference comparison against the official Differential Transformer or
  WDformer code; the equation itself was not covered by an isolated closed-form
  unit check, and the cross-attention (`key_seq != seq`) path was not checked.

## Variants and options

`lambda_init` (re-centres lambda and sets the output scale `1 - lambda_init`) and
`attention_dropout`. The causal variant and the in-component depth schedule of
`lambda_init` are not provided.

## Related components

`self_attention_family` (plain single-softmax attention cores plus the
projecting `AttentionLayer`; this component is the differential alternative
without projections), `topk_expert_attention` (sparse routed alternative),
`wavelet` (the other building block of `wdformer`).
