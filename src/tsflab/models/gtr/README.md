---
name: "GTR"
description: "Global Temporal Retriever: a learned full-cycle memory retrieved by cycle position, fused with the window by 2D convolution, then a residual MLP. Use for periodic multivariate series whose cycle is longer than the lookback; not for aperiodic data or when absolute window positions are unavailable."
---

# GTR

## Idea

- `GlobalTemporalRetriever` holds a trainable `cycle_length x enc_in` matrix (Eqs. 1-5), retrieves the segment at the window's cycle positions (`(start + t) mod cycle_length`) and aligns it with a linear map over time.
- A `[2, P+1]` `Conv2d` mixes the window row with the retrieved reference row per channel, and a residual adds the result back to the input.
- A two-layer GELU MLP with a skip connection and a linear projection produce the horizon per channel (shared weights); `revin` wraps the model.
- The paper presents GTR as a lightweight plug-and-play module that extends a host model's temporal awareness beyond the input window with minimal parameter and compute overhead.

## When to use

- Designed for series with global periodic patterns spanning cycles much longer than the input window (e.g. a weekly cycle seen through a one-day window), as a cheaper alternative to a longer lookback.
- Needs a known cycle length and local period in steps.
- Retrieval needs each window's absolute position: the standard `forward` uses start index 0 for every window, so through the common runner the memory retrieves the same segment every time; use `forecast_at(..., start_index=...)` to get real cycle alignment.
- Not for aperiodic data; channels are processed independently.

## Configure

- `enc_in`: the dataset's channel count.
- `cycle_length`: global cycle in steps (preset 168, weekly on hourly data); must be at least `seq_len`.
- `local_period`: local period in steps (preset 24, daily on hourly data); sets the fusion kernel width `1 + 2 * (local_period // 2)`.

Other hyperparameters: preset defaults in `configs/models/GTR.toml`; tune generically.
The preset follows the official `run.py` default width (`d_model` 512).

## Differences

- Through `tsf run`, the cycle position of each window comes from its first step's calendar marks (`marks.cycle_phase`); the official code reads it from the row index.

- Clean-room rewrite of Eqs. (1)-(5) after inspecting the pinned `models/GTR.py`; nothing copied. Retrieval, temporal alignment, 2D convolution, residual, MLP and RevIN path match the disclosed forecast path.
- The common interface has no absolute sample index, so it uses a zero start; callers with that information use `forecast_at(..., start_index=...)`.
- The cycle memory is learned locally; it is not an external historical database.
- Citation: Cao, F., Dai, L., Han, J., Xiong, H. "Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval." ICLR 2026 (arXiv:2602.10847).
