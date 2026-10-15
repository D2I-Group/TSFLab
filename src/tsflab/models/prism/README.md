---
name: "PRISM"
description: "Hierarchical time-frequency MLP: recursively splits the lookback into overlapping halves, decomposes each into Haar bands, routes bands by their statistics, and forecasts each band with its own head. Use for long-context, channel-independent forecasting of multiscale series; not for cross-channel modelling."
---

# PRISM

## Idea

- Time hierarchy (`split_length`, `split_halves`): every internal node bisects its segment of length `L` into halves of length `L // 2 + ov`, so a 336-step context becomes halves of 176 and then 96 steps with `overlap = 8`.
- Frequency hierarchy (`haar_bands`): each half is split into `K = num_components` same-length bands (`K - 1` orthonormal Haar levels synthesized back to the segment length, summing to the segment); a cascaded EMA ladder replaces Haar when the segment is shorter than `2 ** (K - 1)`. The decomposition is fixed and not differentiated.
- Band router (`BandRouter`): six statistics per band feed one two-layer MLP per node; a temperature-0.2 softmax over bands, then a second softmax between the two halves; each half's weighted band sum feeds the child node.
- Forecast: every routed band of every level gets its own `Linear(n_l, 128) -> LeakyReLU -> Linear(128, H)` head; band forecasts are gated by squeeze-and-excitation, summed, and de-normalized with the per-series mean and std.
- `use_last_layer_only` keeps only the root level's 2K bands (set only by the official ETTh1 script).

## When to use

- Series whose structure lives at several time scales and frequency bands at once (local bursts plus slow oscillations): the tree localizes in time, the bands in frequency, and the router picks the informative bands per segment.
- Long contexts (official runs use 336 steps), since every level halves the segment and needs enough steps for the Haar levels.
- Channels are processed independently with shared weights; not for tasks where cross-channel dependence matters.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.
- `tree_depth` and `overlap` follow `seq_len`: every split segment needs at least 3 steps, and with `overlap = 0` every split segment length must be even.
- `num_components` follows `seq_len`: Haar bands need segments of at least `2 ** (K - 1)` steps, otherwise the EMA ladder is used.

Other hyperparameters: preset defaults in `configs/models/PRISM.toml`; tune generically.

## Differences

Independent rewrite of Section 3.2, Fig. 1 and Appendix A.2 after reading the pinned official code (`nerdslab/PRISM`, `a6342da`, no license file); nothing copied. It follows the official forecast path.

- The paper stitches the last level's bands back to the context with a cross-fade and forecasts from that; the code forecasts from routed bands of each level, which is implemented.
- The paper trains with MSE; all official scripts and the preset use MAE.
- Official dead nodes (leaves, and nodes below the root with `use_last_layer_only`) are not built; the forecast is unchanged.
- Ablation filter banks, plotting code, and the unused hybrid feature extractor are not implemented; no shared wavelet or `revin` component is reused (their padding and scaling differ).
