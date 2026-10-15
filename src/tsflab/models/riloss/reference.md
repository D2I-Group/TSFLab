# RILoss — reference

## Differences in detail

- Implementation: independent local rewrite from the paper (Definition 3 / Eq. 8, Algorithm 1 of Appendix B, implementation details of the experiments section) after reading the pinned official code (`shang-xl/RI-Loss`, revision `55a10ae95bb4e7a7e2cd87df33b9748e95dd07b5`): `Dlinear/hsic.py`, `Dlinear/exp/exp_main.py`, `Dlinear/models/DLinear.py` and `Dlinear/scripts/EXP-LongForecasting/Linear/etth1.sh`. The repository has no LICENSE file, so the license is recorded as `NOASSERTION`; nothing was copied or imported, and the forecaster is the catalog's shared `dlinear` component.
- Resolved from the official code: the residual is transposed to `[batch, channels, horizon]`, so each window's HSIC treats its channels as the kernel samples (length-`H` residual vectors).
- Gram matrices are Gaussian with bandwidth 1 (`exp(-||a - b||^2 / 1)`); the estimator is the biased `tr(K H L H) / (n - 1)^2` form (not the U-statistic of Eq. 4); per-window statistics are summed over the batch before the exponential.
- `lambda = 10`, `tau = 1` (also stated in the paper); the observation term is MSE; DLinear uses `kernel_size = 25` and shared (non-individual) projections.
- Noise range: the paper (Algorithm 1 and the method section) samples `eps ~ U(-1, 1)`, while every official training function draws `torch.rand_like`, i.e. `U(0, 1)`; the preset follows the official code (`noise_low = 0`, `noise_high = 1`) and `noise_low = -1` reproduces the paper's statement. Because the Gaussian kernel depends only on distances, the two differ by a factor-two scale of the noise, not by the shift.
- Observation term: the configured criterion (MSE by default and in the paper); the official code also leaves an MAE variant commented in.
- Single channel: HSIC is undefined with one channel (the official expression divides by zero), so the objective raises `ValueError` for one-channel targets such as `features = "MS"`.
- Carriers: only DLinear is provided; the paper also trains Informer, Autoformer, iTransformer and RAFT with the same loss.
- The title's "learnable" refers to the loss being learned through; there are no learnable loss parameters in the paper or the code.
- Checked behaviour: the Gaussian Gram matrix, the HSIC estimator against an explicit trace formula and its zero value under independence-by-construction, the Eq. (8) assembly with fixed noise, the uniform noise range, the DLinear carrier, the objective's use of the configured criterion and its single-channel error, and gradient flow through the HSIC term; no training was run. Reported benchmark numbers are not reproduction claims of this implementation.

## Paper

RI-Loss: A Learnable Residual-Informed Loss for Time Series Forecasting, AAAI 2026 (arXiv 2511.10130, v2 2025-11-27).

## Citation

```bibtex
@inproceedings{wang2026riloss,
  title     = {RI-Loss: A Learnable Residual-Informed Loss for Time Series Forecasting},
  author    = {Wang, Jieting and Shang, Xiaolei and Li, Feijiang and Peng, Furong},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026},
  eprint    = {2511.10130},
  archivePrefix = {arXiv}
}
```
