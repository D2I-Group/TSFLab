# LiNo — reference

## Differences in detail

- Sources read (revision `0e726e34188ea5c7af5d224d65076d5f5ccbac93`): `models/LiNo.py`, `layers/LiNo.py`, `layers/RevIN.py`, `run.py`, `scripts/Multivariate/*/*.sh`; the rewrite follows Eqs. 1-4 and Figure 2 of the paper (arXiv v4). Nothing was copied or imported.
- Resolved from the official code: the Li block's AR coefficients form a shift-invariant depthwise kernel of length `d_model` per variate over a `d_model - 1` zero left-padding (causal with a full receptive field), initialised to `softmax(ones) = 1 / d_model` with zero bias.
- The frequency projection is a complex `nn.Linear` over the `d_model // 2 + 1` rFFT bins with real default initialisation (imaginary parts zero) and inverse rFFT back to `d_model`.
- The channel and final MLPs are Linear-Tanh-Dropout-Linear; both horizon projections per level start with constant weights `1 / d_model`.
- RevIN has no affine parameters (`eps = 1e-5`, biased variance), reproduced by the catalog `revin` with `affine=False`; the inverted embedding is `embed.DataEmbedding_inverted` without time marks.
- Defaults follow `run.py`: `d_model = 512`, `layers = 2`, `dropout = 0.0`.
- The paper concatenates the softmax-weighted channel mean with `N^TF`; the official code uses the block input `R^L`, and this entry follows the code.
- The official inverse rFFT relies on the default length, returning `d_model - 1` steps for odd `d_model`; here the length is explicit (identical for even widths).
- Univariate (`S`) runs use the same model with one variate. The optional per-block prediction return used for the paper's visualisations is not exposed; `Model.block_predictions` returns the normalized per-level forecasts instead.
- Training uses the catalog trainer and loss (the paper trains with MSE and Adam). Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@article{yu2024lino,
  title   = {LiNo: Advancing Recursive Residual Decomposition of Linear and Nonlinear Patterns for Robust Time Series Forecasting},
  author  = {Yu, Guoqi and Li, Yaoming and Guo, Xiaoyu and Wang, Dayu and Liu, Zirui and Wang, Shujun and Yang, Tong},
  journal = {arXiv preprint arXiv:2410.17159},
  year    = {2024}
}
```
