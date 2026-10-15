---
name: "SCAM"
description: "Training method that co-trains a conv reconstruction net to relabel targets and masks overfitted label points, over an ensemble of channel-independent MLP predictors. Use for univariate-style forecasting with noisy or anomalous target labels; not for channel interactions or probabilistic output."
---

# SCAM

## Idea

- `scam_loss` is Eq. (5): with `A = y_rec - y_hat`, `B = y_rec - y` and the gradient-free mask `M = (A B > 0)`, the loss is `|y - y_hat| (1 - M) + 2 min(|A|, |B|) M`; `official` follows the pinned code (squared supervised error plus `A^2 M`).
- `ConvConcatEmbedding` and `CandidateDecoder` form the reconstruction network `g(y; phi)` (Sec. 2.2, App. D): stride-2 convolutions unfolded to the series length, concatenated across layers, and decoded point-wise into `N` pseudo-label candidates.
- `Model.training_loss` (the declared `training_objective`) co-trains `g` and `N` MLP predictors, candidate `n` paired with predictor `n`; inference returns the ensemble mean without `g`.
- `SpectralNormLinear` is the Spectral Norm Regularization of Sec. 3.4 (Eq. 6) on the first predictor layer, one power iteration per call.

## When to use

- Targets contain noisy or anomalous points that a plain MSE/L1 objective would overfit; the mask replaces such label points with reconstructed pseudo-labels.
- Channel-independent MLP predictors with shared weights: suits weakly correlated channels, not data where cross-channel interactions carry the signal.
- Training trains `N` predictors plus the reconstruction net; inference is a cheap MLP ensemble. Point forecasts only.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly.
- Other hyperparameters: preset defaults in `configs/models/SCAM.toml`; tune generically (`rec_width` must be divisible by `candidates`).

## Differences

Independent rewrite of the pinned official code (`SuDIS-ZJU/SCAM@dafe51de`, no license file, `NOASSERTION`); paper/code conflicts are recorded in `card.toml` issues.

- Loss: `loss_form = "official"` (default) follows the code (squared masked supervised term plus `M * (y_hat - y_rec)^2`); `"paper"` gives the L1 Eq. (5).
- Spectral norm follows the code form (no learnable `gamma`, extra `Linear(d/2, d)` after it, power iteration on every call).
- Candidate count, per-candidate predictor copies, and inference averaging come from the official configs (`num_series: 8`, `num_models: 8`); conv-concat layers use the effective `scam.yaml` values (4 layers, multiplier 4, width 128).
- Kept official behaviour: all `N` candidate heads start from identical weights.
- Fixed: the reconstruction is always cropped to the series length, and the spectral-norm buffer is device-agnostic (official code calls `.cuda()`).

Citation: Yang, Zhang, Liang, Lu, Chen, Li, "Not All Data are Good Labels: On the Self-supervised Labeling for Time Series Forecasting", NeurIPS 2025, arXiv:2502.14704.
