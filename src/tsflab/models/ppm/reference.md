# PPM — reference

## Differences in detail

- Paper source: arXiv 2605.23402 (Sec. 3-6, Eqs. 1-16, Appendix B.1, B.4, Table 6).
- Implementation: independent rewrite after reading the pinned official code (`ljl8336/PPM`, revision `6b4206a8c6bd3d298d8bdc58ebc88e5f5f6c13f9`, Apache-2.0): `model_ppm/PPMF/MLP_PPM.py`, `model_ppm/exp/exp_main.py` (`kde_loss`, `mean_loss`, `train`, `vali`, `test`), `run.py`, `data_provider/`, and `scripts/MLP/*.sh`. Nothing was copied or imported.
- Resolved from the official code: per-window standardization with the unbiased standard deviation plus `1e-6`, applied to every channel separately (the MLP acts on the time axis with shared weights).
- Encoder: `Linear(L, D)`, GELU, then `Linear(D, D)` heads for `mu` and `log sigma^2`, with `sigma = exp(log sigma^2 / 2)`.
- Mapping: input `[z, Linear(L, D)(x)]`, followed by `Linear(2D, d_ff)`, GELU, and `Linear(d_ff, H)`.
- Loss: `alpha * KDE-NLL + MSE(sample mean)` with `alpha = 0.1`, `h = 0.3`, a Gaussian kernel, 100 training draws, and the `-25` floor on each window's mean log-likelihood. Inference uses 100 draws.
- Output: a set of empirical quantiles at the configured levels, so catalog point metrics use the median quantile; the paper's MSE and MAE use the sample mean.
- Not implemented: the alternative kernels of `kde_loss` and the conditional backbones of Appendix C.1 (Non-stationary Transformer, iTransformer).
- Training uses the catalog optimizer and schedule (the paper uses Adam with a learning rate halved every epoch, early stopping after 5 epochs, at most 30 epochs).
- Checked behaviour: the KDE-NLL against a direct evaluation of Eq. (8), the per-window floor, the responsibility-weighted gradient of Eq. (15), the sample-mean MSE of Eq. (9), the empirical quantiles against `torch.quantile`, the reparameterized push-forward sampling against a direct reference with the same noise, channel independence, monotone quantile output, the objective of Eq. (10) for `M` and `MS` targets, gradients, and the strict parameter schema. Reported benchmark numbers are not reproduction claims of this implementation.

## Paper

Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting, ICML 2026 (PMLR 306; arXiv 2605.23402, 2026-05).

## Citation

```bibtex
@inproceedings{li2026ppm,
  title     = {Parametric Prior Mapping Framework for Non-stationary Probabilistic Time Series Forecasting},
  author    = {Li, Jinglin and Tan, Jun and Fang, Qi and Gui, Ning},
  booktitle = {Proceedings of the 43rd International Conference on Machine Learning},
  year      = {2026}
}
```
