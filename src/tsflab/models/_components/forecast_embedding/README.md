---
name: "forecast_embedding"
description: "Linear value projection plus a bias-free projection of six raw calendar columns scaled by fixed maxima and shifted by 0.5, then dropout. Use for models consuming the repository's raw six-column timestamp marks; not for positional or conv token embeddings (embed), pre-normalized or other-width marks, or absent marks."
---

# forecast_embedding

## What it does

`ForecastEmbedding(channels, d_model, dropout)` returns
`dropout(Linear(channels -> d_model)(values) + RawCalendarEmbedding(marks))` with

`RawCalendarEmbedding(marks) = Linear(6 -> d_model, bias=False)(marks / [2100, 12, 31, 6, 23, 59] - 0.5)`

so each of the six raw columns `[year, month, day, weekday, hour, minute]` is
scaled by a fixed maximum into roughly `[-0.5, 0.5]` before projection. There is
no position term and no convolutional token embedding.

## When to use

Use `RawCalendarEmbedding` alone when a model adds its own value projection
(as `timesnet` does with `value_embedding` and `embedding_dropout`). Use
`ForecastEmbedding` for models consuming the repository's raw six-column marks at both encoder
and decoder (`x_mark_enc`, `x_mark_dec`) with a plain value projection. Do not use
when positions or circular-conv token embedding are required (`embed`'s
`DataEmbedding`), when marks have another width or are already normalized or
encoded (the fixed scales assume raw calendar values), or when marks may be `None`.

## Interface

Public symbols: `RawCalendarEmbedding`, `ForecastEmbedding`.

- `RawCalendarEmbedding(d_model: int)`: parameter `projection.weight` `[d_model, 6]`. `forward(marks [B, T, 6]) -> [B, T, d_model]`; raises `ValueError` when `marks.ndim != 3` or the last dim is not 6. The scale vector is created on `marks`' dtype/device each call; no buffers.
- `ForecastEmbedding(channels: int, d_model: int, dropout: float)`: submodules `value` (`Linear(channels, d_model)` with bias), `calendar`, `dropout`. `forward(values [B, T, channels], marks [B, T, 6]) -> [B, T, d_model]`; raises `ValueError` for `values.ndim != 3` or when `marks` and `values` differ in batch or time. Marks are required (no `None` path). `values` and `marks` must be on the parameter device and match the parameter dtype after scaling (float32 by default; float64 marks would fail against float32 weights); `channels` must equal `values.shape[-1]` (not checked beyond the `nn.Linear` error), `dropout` in `[0, 1]` (checked by `nn.Dropout`).
- State-dict keys: `value.weight`, `value.bias`, `calendar.projection.weight`. Stateless apart from dropout.
