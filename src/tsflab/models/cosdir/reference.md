# CosDir — reference

## Paper

- **Title**: Beyond Magnitude and Shape: A Direction-Aware Loss for Time Series Forecasting
- **Venue**: arXiv preprint
- **Published**: 2026 (arXiv: 2026-08)
- **arXiv**: https://arxiv.org/abs/2608.01857

## Differences in detail

- Independent rewrite from Eqs. (1)-(3) and Algorithm 1 after reading the pinned official code
  (`seunghan96/cosdir` at `c1ecebb4`, MIT): `losses/tslib_losses.py`, `exp/exp_tslib.py`, `models/DLinear.py`,
  `run_tslib.py`. Nothing was copied or imported.
- Resolved from the official code: the last observed value is the last input step of the target channels;
  the loss is averaged over samples and channels; the learnable log-variances start at zero and are
  optimized with the model; `dir_lambda = 0.5` is the default (also in the paper); the DLinear backbone uses
  moving-average kernel 25 and shared projections whose weights start at `1/seq_len`.
- The paper evaluates CosDir on 15 backbones; this entry fixes the carrier to DLinear.
- The cosine uses the paper's denominator `||a|| ||b|| + eps` (`eps = 1e-8`); the official
  `F.cosine_similarity` clamps each norm at `1e-8`, so the two differ only for near-zero difference vectors.
- The base loss is the configured criterion: `loss = "mse"` (the presets) matches the paper; other criteria
  are an extension.
- The official code validates and early-stops on the combined loss; the catalog trainer uses the configured
  criterion. Configured weight decay also applies to the CosDir-UW log-variances (the official code uses
  Adam without weight decay).
- Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@article{lee2026beyond,
  title   = {Beyond Magnitude and Shape: A Direction-Aware Loss for Time Series Forecasting},
  author  = {Lee, Seunghan and Lee, Jaehoon and Seo, Jun and Kang, Junhyeok and Han, Sangjun and Yoo, Sungdong and Kim, Minjae and Lim, Tae Yoon and Kang, Dongwan and Choi, Hwanil and Lee, Soonyoung and Ahn, Wonbin},
  journal = {arXiv preprint arXiv:2608.01857},
  year    = {2026}
}
```
