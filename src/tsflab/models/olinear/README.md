---
name: "OLinear"
description: "Linear-based forecaster in an orthogonally transformed temporal domain (OrthoTrans) with a normalized softplus channel-mixing matrix (NormLin) and a residual MLP sequence stage. Use for efficient multivariate forecasting with correlated channels; not for exogenous inputs or probabilistic output."
---

# OLinear

## Idea

- Inputs and outputs are projected through `input_basis` and `output_basis`, identity until `set_orthogonal_bases` installs data-adaptive orthogonal matrices (OrthoTrans: eigenvectors that diagonalize the temporal Pearson correlation).
- `NormLin` mixes channels with a softplus-normalized, row-stochastic matrix (the paper's lightweight replacement for self-attention).
- A residual `sequence_mlp` refines each channel embedding before `decoder` maps the flattened sequence to the horizon.
- `revin` (affine) normalizes the input when `use_revin` is set.

## When to use

- Multivariate data whose channels are correlated: NormLin learns one positive mixing matrix over all channels.
- Tight compute budgets: the model is linear layers plus a small MLP; the paper reports NormLin at about half the FLOPs of self-attention.
- Series with strong step-wise temporal correlation, where decorrelating the lookback with training-split bases helps (only if the bases are installed).
- Not when exogenous covariates or calendar marks matter (marks are ignored), or when quantiles are needed (point output only).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count (sizes the `enc_in x enc_in` NormLin matrix).
- `input_basis`, `output_basis`: `seq_len x seq_len` and `pred_len x pred_len` orthogonal bases computed from the training split's temporal correlation; install them with `set_orthogonal_bases()` (orthogonality is checked). The preset uses identity bases.

Other hyperparameters: preset defaults in `configs/models/OLinear.toml`; tune generically.

## Differences

- Independent implementation from Eqs. (2)-(5); the unlicensed repository (`NOASSERTION`) is reference-only, its source was not inspected or copied.
- `input_basis`/`output_basis` implement OrthoTrans, `NormLin.normalized_weight()` the positive row-wise L1-normalized matrix of Eq. (3), and the residual channel/sequence learners CSL and ISL.
- The paper computes both bases once from training-set lag-correlation matrices; the standalone preset cannot, so it serializes identity bases (no OrthoTrans) unless they are installed externally.
- One CSL/ISL block and a direct flattened decoder; the full training harness and paper hyperparameter sweep are not reproduced.
