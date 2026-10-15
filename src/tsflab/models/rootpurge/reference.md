# RootPurge — reference

## Differences in detail

- Implementation: independent rewrite from Section 4.2 (Eq. 3 and footnotes 3-4) of the paper after reading the pinned official code (`Wangzzzzzzzz/RootPurge`, revision `63973fc52d0e09448357b289a1f6bdfc82c77b8c`, no license file, recorded as NOASSERTION): `RootPurge/linear_model/run_forecasting.py` (`BaseLinearModel`, `SpecLinear`, `RootPurgeModel`) and `RootPurge/run_scripts/run_rootpurge_speclin_logC.sh`. Nothing was copied or imported.
- Both maps are bias-free (`G_W(0) = 0`, so the zero purge target equals the official `model_zero` option); instance normalization subtracts only the window mean; the frequency map scales the inverse transform by `(L+H)/L`.
- The residual is detached before `P` (paper footnote 4). With `H < L` the official code samples, per sample and channel, one length-`L` window from the residual zero-padded by `L-H` on both sides, which places the full residual at a uniformly random offset.
- `lambda` is multiplied by `max(L/H, 1) ** order`; higher-order purging re-applies `G_W o P` to its own output.
- Defaults: frequency domain, shared weights, `purge_order = 1`, `purge_lambda = 0.25` (the paper selects `lambda` from {0.125, 0.25, 0.5}).
- The root-seeking term uses the run's configured criterion (MSE in the paper; set `training.loss = "mse"` to match).
- The inverse rFFT is evaluated at the explicit length `L+H`; the official code uses the default `2 * ((L+H)//2)`, which differs by one sample only when `L+H` is odd.
- Weights are initialised uniformly in `+-1/sqrt(fan_in)` with a zero imaginary part, matching the official `nn.Linear(...).to(cfloat)` cast in distribution.
- For `features="MS"` the residual covers every observed channel of the horizon while the root-seeking term scores only the target channel.
- The official training loop (5000 Adam steps with validation every 50 steps, optional L1/L2 weight decay) is replaced by the catalog trainer; the rank-reduction baselines (RRR, DWRR) of Section 4.1 are not part of this entry.
- Checked behaviour: both maps linear and bias-free, the frequency map against an independent complex-matmul filter, mean-centred forecasting, the `P` operator (offset placement, cropping, identity), the `max(L/H, 1) ** order` scaling, the stop-gradient on the residual, higher-order purging, and the training objective against Eq. (3). Reported benchmark numbers are not reproduction claims of this implementation.

## Paper

Characteristic Root Analysis and Regularization for Linear Time Series Forecasting, ICLR 2026 (arXiv 2509.23597, 2025-09).

## Citation

```bibtex
@inproceedings{wang2026rootpurge,
  title     = {Characteristic Root Analysis and Regularization for Linear Time Series Forecasting},
  author    = {Wang, Zheng and Zhang, Kaixuan and Chen, Wanfang and Lu, Xiaonan and Li, Longyuan and Schlagenhauf, Tobias},
  booktitle = {International Conference on Learning Representations},
  year      = {2026}
}
```
