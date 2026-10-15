---
name: "MTSUNMixers"
description: "Mamba-based dual unmixing: each channel is a convex mix of learned time bases and each step a convex mix of per-sample channel bases, with history reconstruction. Use for multivariate data driven by a few shared patterns; not for exogenous inputs or probabilistic output."
---

# MTSUNMixers

## Idea

- Time unmixing (Eqs. 14-15): `Model.channel_coefficients` patches each channel, runs `TimeEncoderBlock` (`Norm -> MambaBlock`) and maps the flattened tokens to `k2` softmax coefficients `S_c` (non-negative, summing to one per channel, Eq. 6).
- Learnable bases `history_bases` (`A_c`, `T x k2`) and `future_bases` (`A_p`, `H x k2`) share `S_c`: `A_c S_c` reconstructs the history and `A_p S_c` predicts the future (Eq. 17).
- Channel unmixing (Eq. 16): `Model.channel_bases` embeds each channel's window as a token and scans channels with `BiMambaBlock` (forward and backward selective-SSM branches, `ReLU` gate) to get per-sample bases `A_t` (`N x k1`).
- Learnable softmax coefficients `S_t` (`k1 x T`) and `S_p` (`k1 x H`) mix `A_t` (Eq. 18); each path pair is concatenated along time and projected (Eqs. 19-20).
- `training_objective` adds `recon_weight * |X - X_history|_1` to the forecasting loss (Eq. 21).

## When to use

- Multivariate data that a few shared temporal patterns and a few channel patterns explain (low effective rank): both paths are convex mixtures of a small set of bases.
- Channel mixing is part of the model (Bi-Mamba scan over channels), so cross-channel structure can be exploited.
- Not for exogenous or calendar inputs (marks are ignored) or quantile output (point forecast only). Many architectural details are inferred (no official code), so treat results as this implementation's, not the paper's.

## Configure

- `enc_in`: must equal the dataset's channel count.
- `patch_len`: non-overlapping patch length over `seq_len`; `ceil(seq_len / patch_len)` patches, the last filled by repeating the last value.

Other hyperparameters: preset defaults in `configs/models/MTSUNMixers.toml`; tune generically.

## Differences

- Independent implementation from arXiv 2411.17770v1; the linked official repository is unavailable, so there is no pinned revision or license.
- Shared components: `mamba.MambaBlock` for the time encoder and `MambaBlock.selective_scan` for both Bi-Mamba directions; the Bi-Mamba wiring is model-local.
- The many unreported choices (bases counts, patch length, widths, normalization, encoder wiring) follow the resolutions listed under the card's issues.
- Eq. (21)'s forecast term uses the configured loss (the paper uses L1, `mae`), with `recon_weight` the ratio `lambda1 / lambda2`; validation and test use the configured criterion on the forecast only.
- Reported benchmark numbers are not reproduction claims of this implementation.
