---
name: "SEMPO"
description: "Lightweight channel-independent patch Transformer with energy-aware spectral masking and mixture-of-prompts key/value tokens, trained from scratch here. Use for compact forecasting with mixed high- and low-energy frequency content; not for zero-shot use (no pretrained weights) or cross-channel dependencies."
---

# SEMPO

## Idea

- The paper proposes a lightweight foundation model pretrained on relatively small data: energy-aware spectral decomposition keeps low-energy but informative frequencies, and small prompt experts adapt the Transformer across datasets.
- `energy_aware_decomposition` splits the rFFT into high- and low-energy bins with a learnable threshold and sigmoid masks and reconstructs the series (deterministic masks, not the paper's stochastic pre-training masks).
- `router` softly mixes `prompt_experts` per patch token; `prompt_kv` turns the mix into extra key/value tokens prepended to the attention context.
- A single Transformer layer over patch tokens and a flatten `head` give the forecast; `revin` wraps the model.

## When to use

- Compact forecasting under tight parameter budgets; the architecture is designed to be small.
- Series whose informative content includes low-energy frequencies that a top-k spectral filter would drop.
- No pretrained checkpoint or two-stage tuning is shipped, so the paper's zero-/few-shot claims do not apply; the model trains from scratch per dataset.
- Channel-independent with shared weights: not for data whose signal lies in channel interactions. Point forecasts only.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly.
- `patch_len` follows `seq_len`: `seq_len` must be divisible by `patch_len`.
- Other hyperparameters: preset defaults in `configs/models/SEMPO.toml`; tune generically (`d_model` divisible by `num_heads`).

## Differences

Written for TSFLab after inspecting `models/SEMPO.py` and `layers/SEMPO_EncDec.py` at the pinned revision `59233d1` (Apache-2.0); no source copied. Follows the EASD partition, MoP routing, and key/value augmentation equations.

- Differentiable deterministic spectral masks instead of the stochastic multi-mask reconstruction objective.
- One compact attention block; no decoder stack.
- No pre-training corpus or weights, and no two-stage frozen-backbone tuning.

Citation: He, Yi, Ma, Zhang, Niu, Pang, "SEMPO: Lightweight Foundation Models for Time Series Forecasting", NeurIPS 2025, arXiv:2510.19710.
