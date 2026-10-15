# periodic_query_bank — reference

## Origin and granularity

Extracted from `tqnet` (commit `b94ed873`, automated intake of TQNet,
TimePro, Gateformer, CANet), where the retrieved window is transposed to
`[B, C, L]` and used as the attention query over the normalized input
(channel aggregation). The docstring says CycleNet's residual cycle uses the
same table-and-offset idea; `cyclenet` does not import this component, so that
is a documented similarity, not an equivalence. Model-local in `tqnet`: how the
start phase is derived from the last input stamp (`_start_phase`: weekday, or
`weekday*24 + hour`, or hour depending on `cycle`, plus one step, modulo `cycle`), and what the window is used for.

## Invariants and equivalence evidence

- Contract check (at extraction): the single state-dict key `table` `[period, channels]`
  and zero init, output shape and dtype, every output row against
  `table[(phase + t) % period]` including a negative phase and `length > period`,
  gradients reaching `table`, the four `ValueError` cases (2-D phases, `length < 1`,
  `period < 1`, `channels < 1`), and a seeded regression value of output and gradient.
- A model-level check (pre-consolidation suite) confirmed shape and wrap-around rows
  for `period=4, channels=2`, phases `[3, 0]`, length 3.
- A `tqnet` forward check (pre-consolidation suite) covered it as a consumer,
  confirming the temporal query is used as the attention query. No pre-extraction
  `tqnet` reference exists, so state-dict and output equivalence with the
  pre-refactor model is not tested.

## Variants and options

None. There is no random or non-zero initialization, no per-channel period, and
no interpolation for fractional phases.

## Related components

`dominant_periods` (discovers periods instead of assuming one), `marks`
(calendar features from which a start phase can be computed),
`positional_encoding` (position tables indexed by token position, not by calendar phase),
`periodic_alibi_bias` (periodic structure applied as an additive attention bias, no learned
vectors), `deviation_memory` (a learned bank queried by content with attention, not by index).
`cyclenet` keeps its own local cycle table and does not use this component (same indexing
idea, not shared code).
