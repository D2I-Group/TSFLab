# embed — reference

## Origin and granularity

The layer family follows the Time-Series-Library `layers/Embed.py` layout (the
attribute name `tokenConv` and the `timeF`/`fixed` modes come from there).
History records the file only from the initial commit (`0b1fbf2e`) and later
moves (`33ea2050` colocated it under `_components`); the exact upstream revision
and per-class provenance are not recorded. It is cut as one file because these
layers are always combined by the encoder-decoder transformers; what stays
model-local is everything after the embedding (attention stacks, heads) and any
patch scheme that differs from this one (for example `gateformer` builds its
own patch projection with the `positional_encoding` component instead of
`PatchEmbedding`). Consumers: `transformer`, `informer`, `fredf` (via
`DataEmbedding`) and `gpht`, `penguin`, `sensorformer`, `timeexpert` (via
`PatchEmbedding`); the catalog derives the current list from the code.

## Invariants and equivalence evidence

- Shape, state-dict key, invariant, gradient-flow and seeded numerical-regression checks for every public symbol (`DataEmbedding`, `PatchEmbedding`, `PositionalEmbedding`, `TemporalEmbedding`, `TokenEmbedding`), with seeded outputs pinned as regression values.
- The same checks asserted that `DataEmbedding(3, 8, "timeF")` on `[2, 10, 3]` values and `[2, 10, 6]` marks gives `[2, 10, 8]`; `DataEmbedding_inverted(10, 8)` gives `[2, 3, 8]` without marks and `[2, 9, 8]` with six mark channels; `PatchEmbedding(8, 4, 2, 2, 0.)` on `[2, 3, 10]` gives `[6, 5, 8]` with `n_vars=3`; the `wo_pos` variant differs from `DataEmbedding` exactly by the positional term; out-of-range calendar indices raise `IndexError`. Consumers (`transformer`, `informer`, `fredf`, `gpht`, `penguin`, `sensorformer`, `timeexpert`) exercise it indirectly.
- These checks passed in the full suite run of 2026-10-03 before the test suite was consolidated.

## Variants and options

- `embed_type="timeF"` (continuous `Linear` over marks; the default in the Transformer and Informer configs with `freq="h"`), `"fixed"` (frozen sinusoidal), anything else (learned tables).
- `freq="t"` adds a minute table in `TemporalEmbedding`; `freq` has no other effect.
- Quirks worth knowing: `DataEmbedding_inverted.c_in` is the lookback length; `TimeFeatureEmbedding` always uses six input columns unless `input_dim` is given (`freq` and `embed_type` are accepted but unused). The dead `require_grad` typo assignments and the string version comparison were removed; state-dict keys are unchanged.
- `adapt_tslib_marks` in `marks` converts the six-column marks to the five-column categorical layout (or the four-feature hourly `timeF` width); `fredf` calls it before `DataEmbedding`, while `transformer` and `informer` do not.

## Related components

`forecast_embedding` (value plus normalized raw-calendar projection), `marks`
(mark adapters, including TSLib adaptation), `positional_encoding` (standalone
sin/cos or learned position tables; `embed.PositionalEmbedding` is the fixed
sinusoid baked into these embeddings), `flatten_forecast_head` (head commonly
placed after patch tokens), `transformer_encdec` and `self_attention_family`
(the encoder-decoder stack these embeddings feed).
