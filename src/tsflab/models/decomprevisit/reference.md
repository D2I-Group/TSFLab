# DecompRevisit — reference

## Paper

- **Title**: Revisiting the Seasonal Trend Decomposition for Enhanced Time Series Forecasting
- **Venue**: ICASSP 2026
- **Published**: 2026 (arXiv: 2026-02)
- **arXiv**: https://arxiv.org/abs/2602.18465

## Implementation details

- Independent rewrite of the paper's two efficient dual-MLP models (Section 2.5, RMSM and RMM) with the
  moving-average decomposition of Section 2.2 and the trend branch of Section 2.3, after reading the pinned
  official code (`Sanjeev97/Time-Series-Decomposition` at `73f0a4b4`; no license, `NOASSERTION`):
  `models/RMSM.py`, `models/RMM.py`, `layers/MLPs.py`, `layers/Decomposition.py`, `run.py`. Nothing was
  copied or imported. The paper has no single method name; `DecompRevisit` is the catalog name.
- `decompose` is Eqs. (1)-(2); `forecast_trend` is Eq. (8); `ShiftMLP` is Eqs. (10)-(12) (RMSM); the
  `seasonal_head="mlp"` path is RMM, Eq. (13); `forward` is Eq. (14).
- Resolved from the official code: moving-average kernel 25 with edge padding (identical to the cataloged
  `series_decomposition` for odd kernels); RevIN affine with `eps = 1e-5` and population variance (identical
  to the cataloged `revin`); trend MLP hidden width 512; the shift MLP has a 64-unit seasonal embedding and a
  128-unit mixing layer followed by a linear output; RMM applies the trend MLP module itself to the seasonal
  part; the raw (unnormalized) input window is the shift-MLP context.

## Differences in detail

- Eq. (12) writes a ReLU on the shift-MLP output and the setup text lists hidden sizes `[64, 128, 128]`; the
  official code (followed here) has a linear output after the 64- and 128-unit layers, which lets the
  seasonal forecast be negative.
- The model-agnostic variants that use iTransformer, PatchTST, TimesNet, or DLinear without normalization as
  the seasonal backbone (DR- models in the official code) are not part of this entry.
- The mixture-of-experts and FFT decompositions of Section 2.2 (ablations) are not implemented; only the
  moving-average decomposition is. Even moving-average kernels are rejected.
- Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@inproceedings{panta2026revisiting,
  title     = {Revisiting the Seasonal Trend Decomposition for Enhanced Time Series Forecasting},
  author    = {Panta, Sanjeev and Yuan, Xu and Chen, Li and Tzeng, Nian-Feng},
  booktitle = {IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2026}
}
```
