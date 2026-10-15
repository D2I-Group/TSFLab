# transformer_encdec — reference

## Origin and granularity

Present from the initial commit (`0b1fbf2e`) and moved with the other shared
components (`33ea2050`, `02140040`). It was cut separately from the attention
cores (`self_attention_family`) so models choose the attention per role while
sharing the residual/norm/FFN skeleton. `informer` and `transformer` use the full set; `dualformer` uses only
`EncoderLayer`, and `gpht` uses `Encoder` and `EncoderLayer` (no decoder, no distilling). Model-local: embeddings, mask
construction, the generative decoder input, the output projection choice.

## Invariants and equivalence evidence

- Contract checks (at extraction): for `ConvLayer`, the
  state-dict keys (including `norm.running_mean`), the output length
  `(L + 1) // 2 + 1` for `L` in {8, 11}, and finite gradients; for the encoder, the
  `EncoderLayer` key groups, shape, that the
  output is layer-normalized (post-norm), gradients, a 3-layer `Encoder` with a final
  norm, a distilling `Encoder` with two `ConvLayer`s (three attention maps, shortened
  length), the GELU option, and seeded values pinned as a regression value; for the
  decoder, the `DecoderLayer` key groups, shape, that
  later query positions do not change earlier outputs, gradients, `Decoder` with final
  norm and projection (`[2, 5, 3]`), and seeded values pinned as a regression value.
- Model-level checks (pre-consolidation suite) asserted that `transformer` assembles
  full-attention layers in the encoder, decoder self and cross roles (causal
  self-attention only), that `informer` uses ProbSparse attention with
  `len(encoder.conv_layers) == e_layers - 1` for distilling, and that both return
  only the forecast horizon.
- A `dualformer` test constructed the model, which uses `EncoderLayer`.
- No pre-extraction fixture exists, and the `delta` routing is not tested; the
  `activation` and `conv_layers` `ValueError`s were covered by a component
  validation check. These checks passed in the full suite run of 2026-10-03 before
  the test suite was consolidated.

## Variants and options

`conv_layers` turns on Informer distilling; `norm_layer` / `projection` on
`Encoder` / `Decoder` give the final norm and output head; `activation` picks
ReLU or GELU. Pre-norm, RoPE, or parameter-free attention variants are not
supported here.

## Related components

`self_attention_family` (attention cores and `AttentionLayer`; required to supply the
injected attention), `tst_transformer` (the other Transformer encoder component: a
batch-first stack of torch `nn.TransformerEncoderLayer` for patch tokens with no
decoder, no injected attention and no distilling, whereas this one is the injected-attention
Layer API with conv-FFN and a decoder), `masking` (builds the `attn_mask`/`x_mask` inputs).
- `decomposition_encdec`: the Autoformer-style counterpart, with a series decomposition after each sub-layer instead of LayerNorm and a progressive trend output.
