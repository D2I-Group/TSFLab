---
name: "MambaTS"
description: "Mamba over variable-major patch tokens with variable permutation training and a learned (ATSP-solved) variable scan order. Use for long-term multivariate forecasting where cross-variable dependencies matter; not for univariate series or many-channel data where one long cross-variable scan is too costly."
---

# MambaTS

## Idea

- Embeds non-overlapping patches per variable and concatenates tokens variable-major, so a single Mamba scan visits each variable's whole history before the next (VAST).
- `TemporalMambaStack` stacks `mamba` `MambaBlock` mixers with `use_conv=False` (no causal convolution) and `x_dropout` on the step-size/B/C projections, in add-then-LayerNorm order.
- Variable Permutation Training shuffles the variable order per sample each step; `training_objective` credits each sample's loss to its K transitions through `update_scan_costs`.
- `scan_order` solves the asymmetric TSP over the mean transition cost (`anneal_scan_path`, or `greedy_scan_path` for `atsp_solver="GD"`); inference scans in that order.
- `revin` standardises each variable and `flatten_forecast_head` maps each variable's flattened patch tokens to the horizon with one shared linear layer.

## When to use

- Designed for long-term multivariate forecasting where the order in which variables are scanned matters, i.e. there are useful cross-variable dependencies to learn.
- Patching shortens long lookback windows; RevIN handles per-variable level shifts.
- The scan length grows with `enc_in x patches`, so very many channels make it slow (and the sequential PyTorch scan is slower than fused kernels).
- Not for univariate data: the variable-ordering machinery has nothing to order.

## Configure

- `enc_in`: number of variables; each contributes its patch tokens to the one variable-major scan.
- `patch_len`, `stride`: must satisfy `patch_len <= seq_len`; tokens per variable are `(seq_len - patch_len) // stride + 1` with trailing steps dropped (no padding), so choose `patch_len` dividing `seq_len` when `stride == patch_len` (preset 48).

Other hyperparameters: preset defaults in `configs/models/MambaTS.toml`; tune generically.

## Differences

- Independent implementation (official repository MIT; nothing copied); marks and decoder inputs are ignored.
- Sequential PyTorch scan instead of `mamba_ssm`/Triton kernels and `python_tsp`; equivalent but slower.
- Transition costs use the official running mean (Proposition 2), not the paper's Eq. 6 moving average.
- Official per-sample cost attribution bug fixed: each sample's loss goes to its own K transitions; NaN/infinity cases in the official scaling are avoided.
- ATSP: seeded annealing from the greedy path (`SA`) or nearest-neighbour (`GD`); `LS`, `LK`, `Random` are absent; the order is recomputed whenever costs change.
- The head is sized by the true patch count (official width only matches when `stride == patch_len`).
- Full detail in reference.md.
