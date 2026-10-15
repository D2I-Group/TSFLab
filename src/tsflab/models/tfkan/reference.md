# TFKAN — reference

## Paper

TFKAN: Time-Frequency KAN for Long-Term Time Series Forecasting (Kui et al., arXiv:2506.12696, 2025).

## Implementation mapping

- `BSplineKANLinear` (Eqs. 1-4): `W_base SiLU(x)` plus per-edge B-spline coefficients times a per-edge scale on a uniform per-input knot grid over `[-1, 1]` (grid size `s = 2`, order `k = 1` by default), with no output bias; `KAN` stacks two of them through a hidden width.
- Initialization (from the official `models/KAN.py`, `KANLinear.reset_parameters`): Kaiming-uniform base weight and spline scale; spline coefficients fitted by least squares to uniform noise of width `scale_noise / s` at the interior knots.
- `Model.adjust_dimension` (Eq. 5): frequency branch multiplies each channel's series by one learnable `[1, d]` vector (`d = embed_size`).
- `Model.frequency_branch` (Eqs. 6-9): FreqKAN `d -> hidden -> d`, soft-shrinkage threshold `sparsity_threshold`, inverse rFFT with `norm="ortho"` back to length `L`.
- `Model.time_branch` (Eq. 10): TimeKAN `L -> hidden -> L` per channel.
- Predictor (Eq. 12): KAN `L * d -> hidden -> pred_len` on the flattened sum.

## Not reproduced from the official code

- `channel_independence`, `scale = 0.02` and the KAN regularization settings are set but never used upstream; no regularization loss is added.
- The test loader's `drop_last = True` and the `MinMaxScaler` "inverse" scaling are upstream pipeline bugs; TSFLab's evaluator and data pipeline replace them.

## Citation

```bibtex
@article{kui2025tfkan,
  title   = {TFKAN: Time-Frequency KAN for Long-Term Time Series Forecasting},
  author  = {Kui, Xiaoyan and Liu, Canwei and Li, Qinsong and Hu, Zhipeng and Shi, Yangyang and Si, Weixin and Zou, Beiji},
  journal = {arXiv preprint arXiv:2506.12696},
  year    = {2025}
}
```
