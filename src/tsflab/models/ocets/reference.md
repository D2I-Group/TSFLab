# OCETS — reference

## Paper

- **Title**: Beyond MSE: Ordinal Cross-Entropy for Probabilistic Time Series Forecasting
- **Venue**: AAAI 2026 (arXiv 2511.10200, 2025-11)

## Differences in detail

- **Implementation.** Independent rewrite from the Methodology section (Eqs. 2, 8-18) and the Appendix "Data Normalization" (Eqs. 74-77), after reading the pinned official code (`Shi-hm/OCE-TS`, MIT): `models/DLinear.py`, `loss/OrdinalCrossEntropyLoss.py`, `loss/hl_gauss.py`, and `exp/exp_main.py`. Nothing was copied or imported.
- **Resolved from the official code.** The classifier head is the DLinear seasonal/trend projections widened to `pred_len * num_bins` outputs per channel (the paper's Eqs. 13-14 instead describe a DLinear point forecast followed by a separate `W_o` softmax layer, with inconsistent dimensions). Bins are `linspace(low, 1, num_bins + 1)` edges with their midpoints as representatives. The loss sums the cut-point terms and averages over windows, steps and channels, clamping cumulative probabilities to `[1e-8, 1 - 1e-8]` with `1e-8` also inside each logarithm. Defaults `num_bins = 45`, `sigma = 0.015` and moving-average kernel 25 come from `run_longExp.py` and the model file.
- **Normalization.** The lookback is min-max scaled per window and channel along time; the training target is scaled by its own window min/max to build the soft labels; predictions are denormalized with the lookback min/max. The pinned official test loop instead denormalizes predictions with the min/max of the ground-truth future window, which uses future information at inference; that is not reproduced, so the official reported numbers are not comparable to this entry.
- **Value range.** `value_range = "signed"` uses `[-1, 1]` (the official code and the paper's non-ETT setting); `"unit"` uses `[0, 1]` (the paper's ETT setting).
- **Catalog contract.** The declared output is the point forecast (expected bin value), so validation, early stopping and test metrics use the configured point criterion; the categorical distribution is available through `predict_distribution` but is not a catalog `output_type`. The training loss ignores `training.loss`.
- **Not implemented.** The paper's alternative truncated t and Laplace label distributions (appendix) and the NLinear/Linear backbones in the official repository.

## Equivalence checks

The OCE values of the paper's Table 1 are reproduced; Eq. (2) is checked term by term; the truncated-Gaussian bin masses are checked against `math.erf`; the official head layout, the min-max round trip in both value ranges, the training objective, and scale equivariance of the forecast are checked.

## Citation

```bibtex
@inproceedings{wang2026oce,
  title     = {Beyond MSE: Ordinal Cross-Entropy for Probabilistic Time Series Forecasting},
  author    = {Wang, Jieting and Shi, Huimei and Li, Feijiang and Shang, Xiaolei},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```
