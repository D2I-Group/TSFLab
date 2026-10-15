# DB2TransF — reference

## Paper

- **Title**: DB2-TransF: All You Need Is Learnable Daubechies Wavelets for Time Series Forecasting
- **Venue**: arXiv preprint
- **Published**: 2025 (arXiv: 2025-12)
- **arXiv**: https://arxiv.org/abs/2512.10051

## Implementation details

- Independent rewrite from Sections 3.2-3.3 (Eqs. 6-15, Algorithm 1, Figure 1) after reading the pinned
  official code (`SteadySurfdom/DB2-TransF` at `955a6384`; no license, `NOASSERTION`): `model/DB2_TransF.py`,
  `layers/db2vit.py`, `layers/GeneralEncoders.py`, `layers/Embed.py`, `run.py`,
  `scripts/ETT/DB2_TransF_ETTh1.sh`. Nothing was copied or imported.
- `LearnableDB2` is Eqs. (10)-(11): four learnable taps per feature, initialised from db2, give
  `A[n] = sum_k alpha_k x[2n+k]` and `D[n] = sum_k beta_k x[2n+k]` along the token axis. `MultiLevelDB2` is
  Eqs. (12)-(15): `levels` analyses recurse on the approximation. The learnable analysis stays local; the
  fixed-filter `wavelet` and `orthogonal_dwt` components cannot express trainable taps.
- Resolved from the official code: the wavelet runs along the token axis of the inverted embedding (variate
  tokens plus TSLib `timeF` calendar tokens), with `d_model` features split into heads; each level right-pads
  with zeros so the stride-2 four-tap windows fit; the output is the sum of the final approximation and all
  details, each linearly interpolated (`align_corners=False`) back to the token count; the block is
  pre-LayerNorm with residuals around the wavelet mixer plus `Linear(d_model, d_model)` and around a GELU FFN,
  each block followed by `BatchNorm1d(d_model)`; normalization is non-affine instance normalization
  (`eps = 1e-5`); 8 heads and FFN width 768 are fixed officially (exposed as `n_heads`, `d_ff`). Preset sizes
  (`d_model = 256`, `e_layers = 3`, `levels = 1`) follow the official ETTh1 script.

## Differences in detail

- Algorithm 1 concatenates `A_L, D_1..D_L` and trims the residual input; the official code interpolates and
  sums them. This entry follows the code.
- High-pass taps are initialised as the negation of Eq. (7), as in the code; exact db2 values replace the
  official five-digit roundings.
- The instance-normalization scale is detached (catalog `revin`); the official code lets gradients flow
  through the standard deviation.
- Calendar tokens come from raw marks via `marks.adapt_tslib_marks`, hourly `timeF` only (`freq = "h"`);
  `use_marks = false` removes them (the official code always appends them when marks are given).
- Mixed precision and the official learning-rate schedule are run settings, not model behaviour.
- Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@article{gupta2025db2transf,
  title   = {DB2-TransF: All You Need Is Learnable Daubechies Wavelets for Time Series Forecasting},
  author  = {Gupta, Moulik and Tripathi, Achyut Mani},
  journal = {arXiv preprint arXiv:2512.10051},
  year    = {2025}
}
```
