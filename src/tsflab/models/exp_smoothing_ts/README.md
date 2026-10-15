---
name: "ExpSmoothingTS"
description: "Simple exponential smoothing with one learned smoothing coefficient per channel; the forecast repeats the final level. Use as a minimal sanity baseline or for noisy, level-driven series; not for data with trend or seasonality."
---

# ExpSmoothingTS

## Idea

- Updates a level recursively as `alpha * x_t + (1 - alpha) * level`, with a sigmoid-bounded `alpha` learned per channel by gradient descent.
- The forecast is the last level repeated over the horizon: no trend or seasonal state (no Holt-Winters).
- Has `enc_in` parameters in total, the smallest of the neural-wrapped statistical baselines.

## When to use

- A floor baseline: any learned model should beat a flat, exponentially weighted level.
- Fits noisy series with little exploitable structure, and works with very little training data or compute.
- Flat forecasts cannot follow trend or seasonality; avoid on clearly periodic or trending data.

## Configure

- `enc_in`: number of channels (one `alpha` per channel).

Other hyperparameters: preset defaults in `configs/models/ExpSmoothingTS.toml`; tune generically.

## Differences

Independent implementation of the level-only exponential smoothing recurrence; no external source inspected or copied.

- Holt trend and seasonal states are omitted.
- One bounded smoothing coefficient per channel is learned by gradient descent instead of fitted per series.

Citation: Holt, C. C. "Forecasting Seasonals and Trends by Exponentially Weighted Moving Averages." International Journal of Forecasting 20(1), 5-10 (2004). doi:10.1016/j.ijforecast.2003.09.015.
