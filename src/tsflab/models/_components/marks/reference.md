# marks — reference

## Origin and granularity

Added with the PoorOtterBob spatiotemporal/air-quality models (commit `1b0e2f7d`,
"Add six PoorOtterBob models") and the forecasting data-setting modes (`7b7c4704`),
then extended by later refactors (`ef3dd5d2` made the model/data contracts
explicit). The module docstring states the layout follows the BasicTS / LargeST
convention, restricted to two calendar features by user specification; no single
paper is recorded.

The encoder-mark guard `encoder_timef_marks` was extracted (refactor/x1-timefeat) from
six identical model-local methods in `db2_transf` ("DB2-TransF: All You Need Is
Learnable Daubechies Wavelets for Time Series Forecasting"), `karma` ("KARMA: A
Multilevel Decomposition Hybrid Mamba Framework...", WASA 2025), `mambaprobtsf`
("Mamba time series forecasting with uncertainty quantification", MLST),
`minusformer` ("Minusformer: Improving Time Series Forecasting by Progressively
Learning Residuals"), `ssclforecaster` (Park et al., "Self-Supervised Contrastive
Learning for Long-term Forecasting", ICLR 2024) and `vcformer` ("VCformer: Variable
Correlation Transformer...", IJCAI 2024). Appending calendar features as extra variate
tokens was introduced by iTransformer (Liu et al., ICLR 2024); the hourly `timeF`
features are the Informer / Time-Series-Library pipeline (Zhou et al., AAAI 2021).

`days_from_civil` is Howard Hinnant's "chrono-Compatible Low-Level Date Algorithms"
(`days_from_civil`), not a forecasting paper; it was extracted from four model-local
copies in `basisformer` (Ni et al., "BasisFormer", NeurIPS 2023), `emaformer` (Zhang
et al., "EMAformer", AAAI 2026), `mfrs` (Yu et al., "MFRS: A Multi-Frequency Reference
Series Approach...", 2025) and `ssclforecaster` (ICLR 2024), which use it to recover
absolute time from raw marks; `elapsed_minutes` was the identical minute count of
`mfrs` and `ssclforecaster`. It is cut at the data-contract boundary so that node-structured models,
air-quality models, and TSLib transformers read the same marks consistently.
Model-specific handling stays local: how embeddings consume the features, and
any model that needs other calendar features.

## Invariants and equivalence evidence

- Contract checks (at extraction) covered the constants and case-insensitive width
  lookup, the `[0, 1)` range and exact values of the two normalized features, the
  3-D/4-D/`None` branches of `to_spatiotemporal` (zero features for `None`), the
  `to_calendar_spatiotemporal` rejection of a 5-channel covariate,
  `future_time_features` and `coerce_time_length` (truncate, pad by repeating the last
  step, identity returns the same object), and `adapt_tslib_marks` (drop year, pass-through
  of preprocessed widths, hourly `timeF` conversion with the Feb-29 day-of-year value,
  the three `ValueError` cases), plus a seeded regression value for `normalized`,
  `spatiotemporal` and `tslib_hourly`.
- A repository-level check confirmed the shared spatiotemporal adapter's
  `[2, 12, 4, 3]` output and that channel 0 equals the values. The graph-component
  extraction checks ran graph consumers on `to_spatiotemporal` output and used
  `normalized_time_features` in a decoder reference model.
- The TSLib adapter (`adapt_tslib_marks`, `tslib_time_feature_dimension`) is used by `fredf` and `pgn`.
- Contract checks covered the `encoder_timef_marks` guard (disabled, `None`, wrong length, hourly
  conversion) and the civil-date helpers against `datetime` (epoch,
  leap days, century years, pre-1970 dates).
- `encoder_timef_marks` is a verbatim move of the six consumers' method bodies
  (same branches, message and call), so it is equivalent by construction; the methods
  keep their names (`calendar_tokens`, `calendar` in `ssclforecaster`) and delegate.
  It was also pinned against a frozen copy at extraction.
- `days_from_civil` is not verbatim for `basisformer` and `emaformer`, which wrote the
  shifted month as `(m + 9) mod 12` instead of `where(m > 2, m - 3, m + 9)`; the two
  agree for every `m` in `[-9, 14]`, hence for all valid months. `mfrs.civil_days`,
  `ssclforecaster.days_from_civil`, `mfrs.stamp_minutes` and the
  `ssclforecaster.mark_minutes` body are the same expressions. Outputs are exact
  integers, so there are no gradients or tolerances to compare.
  At extraction, frozen copies of both old variants were compared with the
  component exhaustively over years 1600-2500, months 1-12, days
  1-31, and the old minute helpers (including `emaformer._elapsed_minutes` and
  `basisformer.normalized_timestamp`) on float32 marks. No consumer holds parameters
  or buffers from these helpers, so state dicts are unaffected. These frozen-copy
  tests passed in the full suite run of 2026-10-03 before the test suite was
  consolidated.

## Variants and options

- Raw 3-D marks (forecasting datasets) vs. 4-D node covariates (air-quality datasets): same function, branch on `marks.dim()`.
- `to_calendar_spatiotemporal` restricts to 2-channel calendar covariates for models that index embedding tables by time-in-day and day-in-week.
- Time-of-day scale assumes minute-resolution stamps; `weekday` is assumed to be 0 to 6.
- `encoder_timef_marks` fixes `embed_type="timeF"`; models that interpolate or reorder marks
  afterwards (for example `esiformer`) apply that step themselves.
- `emaformer` keeps its local `_elapsed_minutes` (rounds in the input dtype, not float64)
  and `basisformer` its origin-shifted minute count; both now call `days_from_civil`.

## Related components

`embed` (consumes the TSLib-layout marks `adapt_tslib_marks` produces; its
`TemporalEmbedding` expects five columns without year), `forecast_embedding` (embeds raw
six-column marks directly with its own scaling, not the normalized calendar features here),
`graph_utils` (graph supports used by the same consumers).
