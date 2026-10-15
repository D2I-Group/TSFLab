# flatten_forecast_head — reference

## Origin and granularity

Extracted in commit `d38451c3` ("extract flatten forecast head") from the
`patchtst` backbone component (`_components/patchtst`) and from local copies in the
`hdmixer`, `moderntcn`, `timexer`, and `umixer` models, which all used the
PatchTST-style flatten-then-linear head (`origin_models` lists these as the historical
sources; the commit does not name the paper, and PatchTST, Nie et al., ICLR 2023, is
the standard source of that head). Those four models no longer import the head; the
current direct consumers are derived from the code by the catalog, and `quantile_patchtst`
reaches it through the `patchtst` component. The `patchtst` model keeps its own head
code. Cut at the head only: the preceding encoder, the normalization, and any reshape
into `[B, C, D, P]` stay model-local.

## Invariants and equivalence evidence

- Contract check (at extraction, shared and individual modes): the exact state-dict
  keys, the `[2, 3, 5]` output and dtype, gradients to the input and all parameters,
  leading axes preserved in shared mode (`[7, 2, 3, 5]`), and the `RuntimeError` for a
  feature-width mismatch; seeded output values and input gradient for both modes were
  pinned as regression values. Eval mode with `head_dropout=0.5` was checked to be
  deterministic.
- Equation check: shared output equals `linear(flatten(x))`, individual output equals
  the stack of per-channel linears, with `[2, 3, 7]` shapes.
- These checks were part of the pre-consolidation test suite, which passed in full on
  2026-10-03 before the suite was consolidated.
- No frozen pre-extraction copy exists; extraction equivalence for the original
  consumers was never recorded as a check.

## Variants and options

`individual=True` for channel-specific heads (parameters scale with `n_vars`);
`head_dropout` for output dropout. Heads that add a nonlinearity, mix channels, or
project in two stages are not covered and stay local.

## Related components

`channel_wise_linear` (per-channel linear over the time axis of `[B, L, C]`; this head
instead flattens a feature-by-patch grid), `patchtst` (backbone component that composes
this head), `embed` (`PatchEmbedding` produces the tokens such heads read), `dlinear`
(alternative linear forecaster), `quantile_head` and `gaussian_parameter_head`
(probabilistic output heads; this one emits a point forecast).
- `fft_extrapolation_conv`: frequency-domain history-to-horizon alternative.
