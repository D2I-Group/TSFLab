# forecast_embedding — reference

## Origin and granularity

Extracted from an identical model-local calendar-embedding plus forecast-embedding
pair in `autoformer` (Wu et al., "Autoformer: Decomposition Transformers with
Auto-Correlation for Long-Term Series Forecasting", NeurIPS 2021); `fedformer`
(Zhou et al., "FEDformer", ICML 2022) and `dualformer` ("Dualformer:
Time-Frequency Dual Domain Learning for Long-term Time Series Forecasting") consume
the same classes, and `timesnet` (Wu et al., "TimesNet: Temporal 2D-Variation
Modeling for General Time Series Analysis", ICLR 2023) uses `RawCalendarEmbedding`
alone as its calendar term. Which
reference-implementation embedding it stands in for is not recorded, and the
constants (2100, 12, 31, 6, 23, 59) are not tied to any paper: they are the column
ranges of the repository's raw timestamp contract (`[year, month, day, weekday,
hour, minute]`, see `marks`). Cut at the embedding boundary; decomposition,
attention, and heads stay local.

## Invariants and equivalence evidence

- Contract check (at extraction): the state-dict keys (`value.weight`, `value.bias`,
  `calendar.projection.weight`), output shape and dtype, that the calendar term equals
  `projection(marks / [2100, 12, 31, 6, 23, 59] - 0.5)`, gradients on input and all
  parameters, and `ValueError` for marks of width 5, 2-D marks, 2-D values, and
  mismatched time length; seeded output values were pinned as a regression value.
  A second contract check covered the `[2, 5, 8]` output shape and the `ValueError` on
  a time-axis mismatch.
- The regression values were recorded from the extracted class; equivalence with the
  pre-extraction `autoformer` class rests on identical source, not a frozen copy.
- `timesnet` migration (its former model-local `CalendarEmbedding`) is a verbatim
  move: identical body, same submodule name `projection`, same `nn.Linear(6, d_model,
  bias=False)` construction and RNG consumption, so TimesNet's state-dict key
  `calendar_embedding.projection.weight` and its outputs are unchanged by
  construction. Equivalence against a frozen verbatim copy of the old class
  (state-dict keys and values, outputs, gradients equal) was checked at extraction;
  that frozen-copy test passed in the full suite run of 2026-10-03 before the test
  suite was consolidated.

## Variants and options

Only `dropout` is configurable. A zero-dropout instance (`0.0`) is deterministic.
The calendar term cannot be disabled; supply zero marks (output of the calendar
term is then `projection(-0.5 * ones)`, a constant bias-like vector) if no covariates
exist.

## Related components

`embed` (Time-Series-Library style embeddings: also value plus calendar, but with a circular-conv token embedding, positional encoding, and `timeF` mark widths; this one has neither a position term nor a conv and takes raw marks), `marks` (defines the raw six-column layout and mark adapters),
`series_decomposition` (decomposition used by the consuming models).
- `channel_alignment`: channel-width adapter.
