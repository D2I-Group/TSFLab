---
name: "CycleNet"
description: "Residual Cycle Forecasting: subtracts a learnable recurrent cycle aligned by timestamp phase, forecasts the residual with a linear map or small MLP, then adds the cycle back. Use for long-horizon forecasting of strongly periodic series with timestamps; not for aperiodic data or periods the calendar marks cannot index."
---

# CycleNet

## Idea

- `cycle_pattern` is a learnable `[cycle, channels]` parameter (zero-initialized) selected by timestamp phase (hour, weekday, or hour-of-week for cycles 24, 7, 168).
- The aligned cycle is removed from the normalized history and the future cycle is added back after forecasting.
- The residual is forecast by one shared `channel_wise_linear` map, or a small MLP when `model_type='mlp'`.
- Non-affine `revin` is optional; missing timestamps fall back to phase zero.

## When to use

- Series with a stable, known period (daily or weekly cycles in electricity, weather, energy), especially at long horizons.
- Needs calendar marks to align the cycle; without them every window starts at phase zero and the cycle is misaligned.
- Very small parameter count; channels are forecast independently; point output only.

## Configure

- `enc_in`: the dataset's channel count.
- `cycle`: the dominant period in steps. Phase is read from the hour mark (weekday for 7, weekday*24+hour for 168), so 24 and 168 assume hourly data and 7 assumes daily data.

Other hyperparameters: preset defaults in `configs/models/CycleNet.toml`; tune generically.

## Differences

- Paper-driven local implementation; the Apache-2.0 repository is reference only and no source was copied.
- The cycle is aligned from timestamp marks (weekday-aware indices for cycles 7 and 168); missing timestamps use a deterministic zero phase.
- Optional normalization reuses non-affine RevIN.
