---
name: "LIFT"
description: "DLinear forecast refined by FFT-estimated, lag-shifted leading variates through state-gated frequency filters. Use for multivariate data where some channels lead others in time; not for univariate or independent channels (no leaders to exploit)."
---

# LIFT

## Idea

- `Model.estimate_leaders` computes circular cross-correlations of all variate pairs by FFT, keeps lag-local maxima, and picks the top-K leaders with their leading steps and signs (O(L log L) per pair).
- `Model.shift_leaders` shifts each leader by its leading step over the observed window concatenated with the preliminary forecast, and flips negatively correlated leaders.
- `Model.lead_filters` and `Model.refine` generate 2K+1 frequency filters from the leader correlations and a per-variate state distribution, filter the spectra of the forecast, leaders, and leader differences, and mix them with a complex linear map as a residual on the normalized forecast.
- The backbone is the `dlinear` component; the refiner is trained jointly with it.

## When to use

- Designed for multivariate series with lead-lag relations: correlated channels where one variate's past anticipates another's future.
- Leaders are re-estimated per lookback window, so shifting lead-lag relations are tracked online.
- Not useful for univariate data or channels that move independently; the refiner then only adds parameters to DLinear.
- Pairwise correlation costs grow with C^2 (chunked by `lead_chunk_size` to bound memory).

## Configure

- `enc_in`: must equal the channel count; leaders are capped at `min(leader_num, enc_in)`. `seq_len` must be at least 4.

Other hyperparameters: preset defaults in `configs/models/LIFT.toml`; tune generically.

## Differences

- Independent implementation; the official repository (no license file, `NOASSERTION`) was read as reference only.
- The backbone is fixed to the shared DLinear component and trained jointly; official LIFT wraps any backbone and recommends a pretrained, frozen one.
- Leaders, leading steps and correlations are estimated online per normalized lookback window instead of precomputed over the dataset.
- Filter weights follow the official code (an extra constant-one logit), not the paper text.
- Initialisation of the complex mixing map differs; see reference.md.
