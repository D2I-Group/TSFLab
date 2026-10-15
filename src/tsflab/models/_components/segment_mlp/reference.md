# segment_mlp — reference

## Origin and granularity

Introduced by AutoTimes (Liu et al., NeurIPS 2024, "AutoTimes: Autoregressive
Time Series Forecasters via Large Language Models") as the SegmentEmbedding
(Eq. 2-3) and SegmentProjection (Eq. 7) around a frozen LLM. TALON ("Adapting
LLMs to Time Series Forecasting via Temporal Heterogeneity Modeling and
Representation Alignment", arXiv 2508.07195) uses the same map as its
next-segment forecast head (Eq. 13). Both TSFLab models carried an identical
local `SegmentMLP`; it was moved here verbatim. The cut is the map only: the
autoregressive rollout, segment unfolding, and normalization stay model-local.

## Invariants and equivalence evidence

- Verbatim move: the class body (attribute name `layers`, module order, layer
  construction order, defaults, and error message) is identical to the former
  local `SegmentMLP` in `autotimes/model.py` and `talon/model.py` (which differed
  only in the docstring and the local variable name `width`/`widths`). State-dict
  keys, initialization random-number consumption, outputs, and gradients are
  therefore equal by construction; no fixture was needed.
- Component check (at extraction): shapes over arbitrary leading axes,
  state-dict keys, the explicit formula for the linear and the two-layer case,
  rejection of `hidden_layers == 1`, dropout only in training, and gradient
  flow.
- `autotimes` and `talon` model tests exercised the class through the models'
  own imports. These checks passed in the full suite run of 2026-10-03 before
  the test suite was consolidated; the consumers are now checked at admission
  (`tsf model verify`).

## Variants and options

`hidden_layers` (0, or `>= 2`), `hidden_dim`, `dropout`, and `activation` in
`ACTIVATIONS`. The official AutoTimes `MLP` treats `hidden_layers = 1` like 2;
here 1 is rejected (recorded in the AutoTimes card). No normalization, residual
connection, or output activation.

## Related components

`gpt2_backbone` (the LLM trunk these maps surround), `revin` (instance
normalization before segmenting), `embed`, `channel_wise_linear`,
`flatten_forecast_head`, `mixer_block`.
