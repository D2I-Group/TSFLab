---
name: "MambaProbTSF"
description: "S-Mamba mean network plus a softplus MLP sigma network, trained jointly by Gaussian negative log-likelihood. Use for multivariate forecasting that needs a predictive mean and per-step uncertainty (CRPS, intervals); not for heavy-tailed or multimodal targets a Gaussian cannot describe."
---

# MambaProbTSF

## Idea

- Two separate networks (Eq. 1, Fig. 1b): `mean_network` (`SMambaForecaster`) gives `mu_{1:T}(x_{1:P})` and `sigma_net` gives `sigma_{1:T}(x_{1:P})`; output is `[batch, pred_len, channels, 2]` = `(loc, scale)`.
- The mean network is S-Mamba: RevIN, one inverted token per variate (and per hourly calendar feature), blocks that add a forward and a flipped `mamba` scan across tokens, then a feed-forward network, each followed by LayerNorm, and a token-to-horizon projection.
- `SigmaMLP` maps each raw variate history through `Linear(seq_len, 512)`, GELU, `Linear(512, 512)`, GELU, `Linear(512, pred_len)`, softplus plus `1e-8`; `sigma_network = "s_mamba"` swaps in a second S-Mamba (Supplementary A ablation).
- Trained with the Gaussian likelihood of Eqs. (6)-(8) over independent horizon steps (catalog `nll_gaussian`); evaluated with CRPS, coverage, and interval width.

## When to use

- When the task needs a predictive distribution (mean plus scale per step and channel), not only a point forecast.
- Mixes variates through a bidirectional Mamba scan over inverted tokens, suited to correlated multivariate data; RevIN handles level drift in the mean network.
- The output is a single Gaussian per step with independent horizon steps: not for multimodal, skewed, or heavy-tailed predictive distributions, nor for joint sampling of trajectories.
- Calendar tokens exist only for hourly data.

## Configure

- `enc_in`: number of variates; one inverted token and one sigma-MLP pass each.
- `freq`: only hourly (`h`) calendar tokens are supported; set `use_marks = false` for any other sampling rate.

Other hyperparameters: preset defaults in `configs/models/MambaProbTSF.toml`; tune generically.

## Differences

- Independent rewrite from Section 2.2 and Supplementary A after reading the pinned official code (no license file; nothing copied).
- Only the joint (`all_together`) training schedule; the paper's optional mean pre-training stage (`Means_first`) is not expressed by the model contract (emulate by warm-starting).
- The official `sigma_method = Compound` option (not in the paper) is not implemented.
- Pure-PyTorch selective scan instead of `mamba_ssm` fused kernels; numbers may differ slightly.
- The RevIN scale is detached (catalog `revin`); the official code lets gradients flow through it.
- Official training-loop bugs (skipped step every 100th batch, leftover `break` in `run_prob.py`) are not reproduced.
