# WFTNet — reference

## Paper

WFTNet: Exploiting Global and Local Periodicity in Long-term Time Series Forecasting (ICASSP 2024; arXiv 2309.11319, revised 2024-01). Sources used: Sections 2-3 (Eqs. 1-2, WFTBlock and PWC definitions, Fig. 1).

```bibtex
@inproceedings{liu2024wftnet,
  title     = {WFTNet: Exploiting Global and Local Periodicity in Long-term Time Series Forecasting},
  author    = {Liu, Peiyuan and Wu, Beiliang and Li, Naiqi and Dai, Tao and Lei, Fengmao and Bao, Jigang and Jiang, Yong and Xia, Shu-Tao},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2024}
}
```

## Implementation mapping

Independent rewrite after reading the pinned official code (`Hank0626/WFTNet` at `7fc60ceb`, Apache-2.0): `models/Wavelet.py`, `layers/Conv_Blocks.py`, `layers/Embed.py`, `data_provider/data_factory.py`, `exp/exp_main.py`, `run.py`, `scripts/*.sh`, and the `cwt` routine of its `ptwt==0.1.6` dependency. Nothing was copied or imported.

Resolved from the official code:

- CWT: real Morlet `exp(-t^2/2) cos(5t)` sampled on `[-8, 8]` with `2^10` points and rectangle-integrated, resampled per scale, linearly convolved, differentiated, scaled by `-sqrt(s)`, and centre-cropped (pywt-style discretisation), at 8 scales `2 ** linspace(-1, wavelet_scale, 8)`.
- Inception block: kernels `1, 3, ..., 2 * num_kernels - 1`, Kaiming fan-out init, averaged; it and its GELU sandwich are shared between the Fourier maps and the wavelet map. Strip convolution: depthwise, kernel `(8, 1)`.
- PWC exponent `n = 10`; `alpha` from `m = 5` lowest FFT bins including DC.
- Embedding: Time-Series-Library value (circular conv) + sinusoidal position + `timeF` calendar embedding with dropout; one LayerNorm module reused after every block; non-affine mean/std instance normalization.
- Preset: `scripts/ETTh2.sh`, horizon 96 (`d_model = d_ff = 32`, `e_layers = 1`, `top_k = 5`, `wavelet_scale = 8`, `num_kernels = 6`, `dropout = 0.1`).

## Differences in detail

- `alpha` (leakage fix): the official `period_coeff` in `data_provider/data_factory.py` reads the whole CSV (0.7845 for ETTh2). The preset uses 0.8126, the same statistic (raw values, all seven numeric columns, `m = 5`) on the TSFLab training split only: rows 0-8,639 of `ETTh2.csv`, the first 60% of the 14,400 rows the loader reads.
- `top_k = 0`: switches the official block to a wavelet-only variant that also projects time from `T_s + T_p` to `T_p` inside the block; not the paper's WFTBlock, not implemented.
- CWT: tabulated once in float64 and applied in the input dtype; the official index grid is evaluated in the data dtype.
