# TimeMixer — reference

## Paper

TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting (Wang et al., ICLR 2024;
arXiv:2405.14616).

Time series show distinct patterns at different sampling scales: microscopic information at fine
scales and macroscopic information at coarse ones. TimeMixer is a fully MLP architecture with
Past-Decomposable-Mixing (PDM), which decomposes multiscale series and mixes seasonal components
fine-to-coarse and trend components coarse-to-fine, and Future-Multipredictor-Mixing (FMM), which
ensembles predictors over scales. It reports consistent state of the art in long- and short-term
forecasting with favourable run-time efficiency.

## Implementation notes

- Multiscale observations: average pooling with window and stride `down_sampling_window`, applied `down_sampling_layers` times.
- The moving-average split is equivalent to replicate-padding the series by `(moving_avg - 1) // 2` on each side.
- `DFTDecomposition` keeps the `top_k` largest non-DC frequencies as seasonal; the remainder is trend.
- Each scale has its own affine RevIN; the summed forecast is denormalized with the finest-scale statistics, which is why `enc_in` must equal `c_out`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/WangWSHLMZ024,
  author    = {Shiyu Wang and Haixu Wu and Xiaoming Shi and Tengge Hu and Huakun Luo and Lintao Ma and James Y. Zhang and Jun Zhou},
  title     = {TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting},
  booktitle = {The Twelfth International Conference on Learning Representations, {ICLR} 2024},
  publisher = {OpenReview.net},
  year      = {2024},
  url       = {https://openreview.net/forum?id=7oLshfEIC2}
}
```
