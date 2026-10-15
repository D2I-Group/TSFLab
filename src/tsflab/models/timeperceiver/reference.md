# TimePerceiver — reference

## Paper

TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting (Lee, Lee;
NeurIPS 2025; arXiv:2512.22550).

Prior forecasting work focuses on encoder design and treats decoding and training as secondary.
TimePerceiver generalizes forecasting to extrapolation, interpolation and imputation with arbitrarily
positioned input and target segments, and pairs it with an encoder-decoder: latent bottleneck
representations interact with all input segments to capture temporal and cross-channel dependencies,
and learnable queries for target timestamps retrieve the relevant information. It consistently and
significantly outperforms prior baselines on a wide range of benchmarks.

## Implementation notes

- Patches: `ceil(seq_len / patch_len)` non-overlapping patches per channel, embedded by `Linear(patch_len, d_model)`.
- Encoder, latent blocks and decoder are attention blocks with LayerNorm and a GELU feed-forward; `latent_dim` must be divisible by `n_heads`.
- Time features: from the last `pred_len` steps of `x_mark_dec`; raw 6-field stamps give month/12, day/31, weekday/7, hour/24, other marks are truncated or zero-padded to 4 features. Without marks, `[t, t^2, sin 2πt, cos 2πt]` over `t in [0, 1]` is used. The features are projected to `latent_dim` and added to the target queries.

## Citation

```bibtex
@misc{lee2025timeperceiver,
  author        = {Jaebin Lee and Hankook Lee},
  title         = {TimePerceiver: An Encoder-Decoder Framework for Generalized Time-Series Forecasting},
  year          = {2025},
  eprint        = {2512.22550},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2512.22550}
}
```
