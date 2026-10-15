# OLMA — reference

## Differences in detail

Checked the paper (Sec. 3.2, Eq. 11-15, Appendix A.3-A.4 hyperparameters and the OLMA+MSE comparison) and, at the pinned revision, `OLMA.py` (the only code file; the repository ships the loss, not the training pipelines). `forward` reads only `x_enc` `[B, seq_len, enc_in]` and returns the `[B, pred_len, enc_in]` point forecast; validation and test metrics use the configured criterion on that forecast, as the training-objective contract prescribes.

- **Carrier.** The paper applies OLMA to TimeCMA, iTransformer, TimesNet, DLinear, TimeMixer, TimeXer, S-Mamba and WPMixer. This entry provides one carrier, DLinear (the `dlinear` component with shared projections, kernel 25), which the paper also uses for its frequency-bias analysis (Sec. 4.2). Other backbones would need the same objective attached to them.
- **Reduction.** The paper writes summed L1 norms (Eq. 11, 13); the official code averages each term over its coefficients (`mean |.|`). The official means are used, which changes the absolute scale of each term but not the weighted form.
- **Channel DFT.** Eq. (12) lists all `c` bins; the official code uses the real FFT (non-redundant `c // 2 + 1` bins), which is reproduced. Likewise the temporal DFT uses the real FFT over the horizon.
- **Wavelet.** Only the Haar wavelet of Eq. (14) is provided; the official helper also offers db2/sym2/db4/coif1/bior1.3 filters, not used in the paper's experiments. `haar_dwt1d` replicate-pads an odd-length horizon where the official code zero-pads; the L1 error summed over the final approximation/detail pair is identical in both cases (`sqrt(2) |e|`), so the loss is unchanged. The detail sign convention also does not affect the L1 error.
- **Weights.** Defaults `(alpha, beta, gamma) = (0.34, 0.33, 0.33)` as in the paper and code; the paper uses `(0.1, 0.45, 0.45)` for ECL and Traffic. The paper states `alpha + beta + gamma = 1`; like the official function this is not enforced, and `loss_weight` is the official overall multiplier.
- **Objective.** OLMA replaces the time-domain loss, as in the paper's main tables; the OLMA + MSE variant of Appendix A.4 is not provided. With `features="MS"` the loss is computed on the single target channel, so the channel term degenerates to the absolute error of that channel.
- **Checks.** Each loss term was checked against the paper's equations and the official reference formulas.

## Citation

```bibtex
@misc{shi2025olma,
  author        = {Tianyi Shi and Zhu Meng and Yue Chen and Siyang Zheng and Fei Su and Jin Huang and Changrui Ren and Zhicheng Zhao},
  title         = {OLMA: One Loss for More Accurate Time Series Forecasting},
  year          = {2025},
  eprint        = {2505.11567},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
