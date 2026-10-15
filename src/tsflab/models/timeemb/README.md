---
name: "TimeEmb"
description: "Static-dynamic disentanglement: a learned per-hour (and optionally per-weekday) complex spectrum bank is the static part, an input-conditioned full-spectrum filter handles the dynamic rest, then an MLP. Use for calendar-driven series under distribution shift; not for data without timestamps or cross-channel structure."
---

# TimeEmb

## Idea

- `GlobalCalendarEmbedding` stores a learnable complex rFFT spectrum per channel for each hour-of-day (and optionally day-of-week) slot, looked up from the first forecast step's timestamp mark.
- The static spectrum is subtracted from the input spectrum; `DynamicSpectrumFilter` rescales the residual by `1 + gate * response`, with the gate conditioned on mean spectral energy; the static part is then added back.
- Inverse FFT gives a filtered lookback that a two-layer MLP maps to the horizon, inside `revin`.

## When to use

- Non-stationary series with strong daily (and weekly) rhythms: the time-invariant calendar component is learned globally and separated from short-term fluctuations.
- Lightweight settings: the model is an FFT, an embedding lookup, a small gate and a two-layer MLP.
- Channel-independent MLP head with per-channel calendar spectra; not for data without timestamps (falls back to one zero slot) or where cross-channel interactions dominate.

## Configure

- `enc_in`: number of channels (each slot stores one spectrum per channel).
- `hour_length`: number of hour-of-day slots; the index is the hour mark modulo `hour_length` (24 = hourly slots; sub-hourly data shares the hour's slot).
- `day_length`: number of day-of-week slots (7); used only with `use_day_index = true`.

Other hyperparameters: preset defaults in `configs/models/TimeEmb.toml`; tune generically.

## Differences

- Clean-room implementation from the paper's global static spectrum embedding and input-conditioned full-spectrum dynamic filter; the author repository (`showmeon/TimeEmb` at `9adf3fba801b34642e7191b45e08aff224b26e67`) declares no license and was not copied.
- The hour and day indices of the first forecast step come from TSFLab decoder marks (encoder marks' last step if absent); calendar marks are optional and fall back to a deterministic zero slot.
- Disabled embedding tables are not registered as dead trainable parameters.
- Plug-in integrations with other forecasters, published training settings, and reported results are not reproduced.
