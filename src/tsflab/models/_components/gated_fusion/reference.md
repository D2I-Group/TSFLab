# gated_fusion — reference

## Origin and granularity

Added in commit `b94ed873` (automated intake of TQNet, TimePro, Gateformer, CANet)
for `gateformer`, which uses it twice: fusing a global (inverted) variate embedding
with a temporal-patch embedding, and fusing that result with a cross-variate encoder
output. The module docstring says several forecasters share this "global/local"
gate; only `gateformer` consumes it today, and no pre-extraction reference exists.
Cut at the fusion only: the branches feeding `a` and `b` and their encoders remain
model-local. Other gates in the repository are different operations and stay local
(`awemixer`'s `CoherentGatedFusion` gates a residual cross-attention context;
`canet`'s `StyleBlendingGate` blends statistics with a fixed ratio).

## Invariants and equivalence evidence

- Contract check (at extraction): state-dict keys, output shape and dtype, every output element lies between the
  two inputs, gradients for both inputs and all parameters, 2-D input, `ValueError`
  for mismatched shapes and for `dim = 0`; seeded output and input gradients were
  pinned as regression values.
- Interpolation check: with both gate weights zeroed, the `a` gate bias at 10 and the
  `b` gate bias at 0, the output approaches `a`. A `gateformer` forward/backward
  check covered the composed pass. These passed in the full suite run of
  2026-10-03 before the test suite was consolidated.
- No reference from before the extraction and none against official Gateformer code.

## Variants and options

Only `dim`. Not covered: gates of different widths or inputs, scalar (per-position)
gates, a residual form `a + g * b`, or gating on concatenated inputs.

## Related components

`harmonic_energy_gate` (a non-learned, spectral-energy gate for blending two
branches), `softmax_gate` (feature gate on a single stream, no second input),
`mixer_block` (residual mixing within one stream), `flatten_forecast_head` (used by
`gateformer` to produce the temporal branch).
- `gated_dilated_conv`: tanh x sigmoid temporal unit, not a blend of two tensors.
