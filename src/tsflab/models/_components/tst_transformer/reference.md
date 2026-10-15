# tst_transformer — reference

## Origin and granularity

The original PatchTST code carried a hand-written `TSTEncoder`; it was replaced by this
compact torch-native encoder with the same role. It is cut as its own component because
it is the only transformer part of the PatchTST backbone and is reusable by any
channel-independent patch or token model. Consumers: the `patchtst` component, the
`stdmae` model (final `LayerNorm`), and the `composed` model through its `temporal` slot
adapter (`models/_slots/adapters.py`, which also serves `component:patchtst`). It is not a
numerical copy of the original: the BatchNorm choice only affects the *final* norm (the
inner layers always use torch's LayerNorm), residual attention and custom `d_k`/`d_v` are
dropped. Embeddings, patching, and heads stay in the consumer.

## Invariants and equivalence evidence

- Contract checks (at extraction), for BatchNorm/post-norm and LayerNorm/pre-norm,
  plus a validation check, covered:
  state-dict keys `layers.layers.<i>.*` and `layers.norm.*`, `running_mean` present only for
  BatchNorm, output shape and dtype, extra `forward` keywords ignored, finite input gradient
  and parameter gradients in training mode, and `ValueError` for indivisible `d_model`,
  `d_k=3`, `activation="swish"`, and `norm="rms"` (extra constructor keywords and a
  callable activation accepted). Seeded eval-mode outputs for both norm choices were
  pinned as regression values.
- A repository contract check asserted the dependency closure of
  `patchtst` contains `tst_transformer`.
- A probabilistic-forecaster test exercised it through
  `QuantilePatchTST` (forward only; shape and quantile contracts).
- These checks passed in the full suite run of 2026-10-03 before the test suite was
  consolidated.
- No fixture compares this encoder with the original PatchTST `TSTEncoder`, and the
  rewrite is known not to be state-dict compatible with it.

## Variants and options

`pre_norm=True` for pre-norm layers; `norm="LayerNorm"` to avoid batch statistics
(`stdmae` uses it); `activation` as relu/gelu/callable. Per-layer
BatchNorm and residual attention are not available.

## Related components

`patchtst` (consumer: the full backbone that wraps this encoder with RevIN, patching, and a head),
`positional_encoding` (add before encoding), `transformer_encdec` (the other Transformer
encoder stack: post-norm with conv-FFN and an injected attention core, versus this fused torch
layer stack), `self_attention_family` (swappable attention cores this encoder cannot take).
