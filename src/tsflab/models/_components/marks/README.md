---
name: "marks"
description: "Adapters from raw six-column timestamp marks to normalized calendar features, spatiotemporal node tensors, Time-Series-Library timeF layouts, and civil-date day and minute counts. Use for models where timestamps carry signal; not for year/month features, other mark layouts, or sub-minute sampling."
---

# marks

## What it does

Pure tensor helpers (no parameters) that turn the framework's decoder-style
4-tuple marks into what model families expect.

- Raw marks are `[year, month, day, weekday, hour, minute]` per step.
- `normalized_time_features`: `time_in_day = (hour * 60 + minute) / 1440` and `day_in_week = weekday / 7`, stacked on the last axis.
- `to_spatiotemporal(values, marks)`: `cat([values[..., None], feats], -1)` where `feats` is the 4-D `marks` as given, or the two normalized features broadcast over the `N` nodes.
- `to_calendar_spatiotemporal`: same, but 4-D covariates must have exactly 2 channels.
- `future_time_features(marks, n)`: only the `feats` part, `[B, T, N, F]`.
- `coerce_time_length(marks, length)`: keep the last `length` steps, or repeat the last step to pad.
- `adapt_tslib_marks`: convert raw marks to the Time-Series-Library layout.
- `encoder_timef_marks(marks, *, seq_len, freq, enabled)`: the encoder-window guard of
  inverted-token forecasters: `None` when disabled or absent, else checks
  `[B, seq_len, k]` and returns `adapt_tslib_marks(marks, embed_type="timeF", freq)`.
- `days_from_civil(y, m, d)`: proleptic Gregorian date to days since 1970-01-01,
  `era * 146097 + doe - 719468` with `y' = y - [m <= 2]`, `era = floor(y' / 400)`,
  `yoe = y' - 400 era`, `doy = floor((153 ((m + 9) mod 12) + 2) / 5) + d - 1`,
  `doe = 365 yoe + floor(yoe / 4) - floor(yoe / 100) + doy`.
- `cycle_phase(marks, cycle, step)`: the position of window step `step` in a calendar cycle:
  weekday (cycle 7), hour of the week (168), steps since midnight for a cycle dividing one day
  (24, 96, 144), steps since Monday for one dividing a week, otherwise the hour; zero without marks.
- `elapsed_minutes(marks)`: `1440 * days_from_civil(year, month, day) + 60 hour + minute`
  on float64-rounded int64 marks.

## When to use

Use when timestamps carry signal (daily and weekly cycles, calendar effects) and
a model needs them in its own layout: the normalized time-of-day/day-of-week
helpers feed graph, spatiotemporal and air-quality models and future covariate
blocks; `encoder_timef_marks` gives inverted-token models hourly `timeF` calendar
tokens; `days_from_civil` / `elapsed_minutes` give an absolute time index (phase,
window position) from raw marks. Do not use when a model needs year, month, or
day features (discarded here), when marks have a different column layout or are
already normalized or TSLib-preprocessed, or when sampling is finer than one
minute. Most functions assume the layout and do not validate it.

## Interface

Public symbols: `TIME_FEATURES`, `TSLIB_TIME_FEATURE_DIMS`,
`tslib_time_feature_dimension`, `adapt_tslib_marks`, `encoder_timef_marks`,
`days_from_civil`, `elapsed_minutes`, `normalized_time_features`,
`to_spatiotemporal`, `to_calendar_spatiotemporal`, `future_time_features`,
`coerce_time_length`. All are stateless tensor ops on float tensors; none has parameters.

- `TIME_FEATURES = 2`: count of normalized calendar channels.
- `TSLIB_TIME_FEATURE_DIMS`: `{"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}`, the pinned TSLib `timeF` widths.
- `tslib_time_feature_dimension(freq)`: case-insensitive lookup; raises `ValueError` listing supported keys.
- `adapt_tslib_marks(marks, *, embed_type, freq)`: `None` returns `None`; `marks.ndim != 3` raises `ValueError`. For `embed_type != "timeF"`: width 6 returns `marks[..., 1:]` (drops year), width 5 passes through, otherwise `ValueError`. For `"timeF"`: width equal to the pinned width passes through; width 6 with `freq == "h"` is converted to `[hour/23 - 0.5, weekday/6 - 0.5, (day-1)/30 - 0.5, (day_of_year-1)/365 - 0.5]` (leap-year aware); anything else raises `ValueError` (only hourly can be reconstructed from raw marks).
- `encoder_timef_marks(marks, *, seq_len, freq, enabled=True)`: returns `None` when `enabled` is false or `marks is None`; raises `ValueError("x_mark_enc must be [batch, seq_len, mark_columns]")` unless `marks` is 3-D with `shape[1] == seq_len`; otherwise `adapt_tslib_marks(marks, embed_type="timeF", freq=freq)` (so `[B, seq_len, 4]` for hourly raw marks).
- `days_from_civil(year, month, day) -> int64`: elementwise on broadcastable integer tensors (int64 expected; `month` 1-12). Exact integer arithmetic with floor division, valid for negative years; day 0 is 1970-01-01. No validation of day or month ranges.
- `elapsed_minutes(marks [..., 6]) -> int64 [...]`: `marks.double().round().long()`, then `days * 1440 + hour * 60 + minute`; the weekday column is ignored and the width is not checked (callers that need a check, such as `ssclforecaster.mark_minutes`, validate first).
- `normalized_time_features(marks [B, T, 6]) -> [B, T, 2]`: reads columns 3 (weekday), 4 (hour), 5 (minute); no shape validation, values are not clamped.
- `to_spatiotemporal(values [B, T, N], marks) -> [B, T, N, 1 + F]`: `marks` 4-D is used unchecked; `None` is replaced by zeros `[B, T, 6]` (so both features become 0); otherwise `marks` is read as `[B, T, 6]`.
- `to_calendar_spatiotemporal(values, marks)`: raises `ValueError` if `marks` is 4-D with last dim != 2 (embedding-index models would otherwise index out of range), else delegates.
- `future_time_features(marks, n) -> [B, T, N, F]`: 4-D returned as-is; else normalized features expanded to `n` nodes (expand, not copy). `marks=None` is not handled (raises).
- `coerce_time_length(marks, length)`: adjusts dim 1 only; works for raw or node-structured marks. Padding repeats the last step; an empty time axis is not handled.
