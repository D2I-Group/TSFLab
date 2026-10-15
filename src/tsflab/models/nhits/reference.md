# NHiTS — reference

## Paper

- **Title**: N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting
- **Venue**: AAAI 2023 (arXiv 2201.12886, 2022-01)
- **Abstract (shortened)**: Long-horizon forecasting suffers from volatile predictions and high computational cost. N-HiTS combines hierarchical interpolation with multi-rate data sampling, so it assembles predictions sequentially, emphasizing components with different frequencies and scales while decomposing the input and synthesizing the forecast. The authors prove that hierarchical interpolation efficiently approximates arbitrarily long horizons in the presence of smoothness, and report almost 20% average accuracy improvement over recent Transformers while cutting computation time by about 50x.

## Implementation mapping

- Multi-rate sampling: each block left-pads the input (replicate) to a multiple of its kernel, then applies max or average pooling with stride equal to the kernel.
- Coefficients: an MLP (`mlp_units`, `activation`, `dropout`) maps the pooled input to a full-length backcast and `ceil(pred_len / n_freq_downsample)` forecast knots; knots are upsampled with `F.interpolate`.
- Hierarchy: residual = residual - backcast between blocks; the last block's backcast layer is frozen because it is unused; forecasts are summed on top of the last observed value.
- Stack lists (`stack_types`, `n_blocks`, `n_pool_kernel_size`, `n_freq_downsample`) must have equal length; `mlp_units` is either one entry shared by all stacks or one per stack.

## Citation

```bibtex
@misc{challu2022nhits,
  author        = {Cristian Challu and Kin G. Olivares and Boris N. Oreshkin and Federico Garza and Max Mergenthaler-Canseco and Artur Dubrawski},
  title         = {N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting},
  year          = {2022},
  eprint        = {2201.12886},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2201.12886}
}
```
