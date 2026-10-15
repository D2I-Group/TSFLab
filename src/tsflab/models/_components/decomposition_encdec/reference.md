# decomposition_encdec — reference

## Origin and granularity

Autoformer (Wu et al., NeurIPS 2021) introduced the decomposition encoder
(Eq. 3) and decoder with progressive trend accumulation (Eq. 4); FEDformer
(Zhou et al., ICML 2022, "FEDformer: Frequency Enhanced Decomposed Transformer
for Long-term Series Forecasting") keeps the same layers and replaces
auto-correlation by the Fourier-enhanced block and attention. The clean-room
`autoformer` and `fedformer` models carried identical copies of the scaffold
(and of its feed-forward helper) that differed only in the injected mixers and
their attribute names; they were extracted together. The cut is one layer: the
mixers, the embeddings, the decoder initialization (seasonal and mean trend), the
final LayerNorm and the seasonal projection stay model-local. The trend
projection here is a bias-free `nn.Linear` per extracted trend, as in both local
implementations (the official Autoformer code uses one circular `Conv1d` of
kernel 3 on the summed trend).

## Invariants and equivalence evidence

- Equivalence against frozen pre-extraction copies of `AutoformerEncoderLayer`,
  `AutoformerDecoderLayer`, `FEDformerEncoderLayer`, `FEDformerDecoderLayer` and
  `_feed_forward` was checked at extraction: old and new encoder and decoder
  layers built from the same seed have identical `state_dict()` key order and
  values, outputs, input gradients and parameter gradients. The same check
  verified that the full models keep their layer keys (`correlation`,
  `self_correlation`, `cross_correlation`, `frequency_block`, `self_frequency`,
  `cross_frequency`) and that no `mixer` key appears, and checked the layer
  equations with stub mixers.
- The migration is not a verbatim move (the mixers are now injected and named by
  argument), so equivalence rests on that frozen-copy check. No reference file is
  stored; the frozen-copy test passed in the full suite run of 2026-10-03 before
  the test suite was consolidated.
- A model-level check in the pre-consolidation suite built
  `AutoformerDecoderLayer` through its unchanged model-local signature.

## Variants and options

- `mixer_name` / `mixer_names`: attribute names only, chosen so consumers keep
  their checkpoint keys; they do not change the computation.
- `activation`: `"gelu"`, anything else gives ReLU (the local behaviour of both
  consumers, kept as is).
- Not covered: a learnable trend projection by convolution (official Autoformer),
  LayerNorm variants (`my_Layernorm`), mixers that return attention maps, or
  decoders without trend output.

## Related components

- `series_decomposition`: the moving-average split used after every sub-layer.
- `transformer_encdec`: post-norm Transformer encoder/decoder layers with an
  injected attention and no decomposition.
- `forecast_embedding`: the value and calendar embedding both consumers place in
  front of these layers.
