---
name: "QuantilePatchTST"
description: "PatchTST backbone (patch tokens, channel-independent Transformer, RevIN) with a monotone quantile head trained by pinball loss. Use for probabilistic forecasts on long lookbacks; not when channel interactions matter or only point forecasts are needed."
---

# QuantilePatchTST

## Idea

- `PatchTSTBackbone` (patch tokens, channel-independent Transformer encoder, flatten head, RevIN) produces the point forecast.
- `QuantileHead` turns each point into `Q` quantiles from a median anchor with cumulative softplus offsets, so quantiles cannot cross.
- `output_type = "quantile"` returns `(B, pred_len, C, Q)`; pair it with `[training] loss = "quantile"` (pinball). `quantile_levels` are injected from `evaluation.quantile_levels`.
- The head and pinball protocol are TSFLab additions, not part of the PatchTST paper.

## When to use

- Tasks that need quantile forecasts with a strong Transformer backbone; patching lets the encoder use long lookbacks at moderate cost.
- RevIN absorbs per-window level and scale shifts.
- Not when cross-channel dependence carries the signal: channels share one encoder and never interact.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.

Other hyperparameters: preset defaults in `configs/models/QuantilePatchTST.toml`; tune generically.

## Differences

Independently composed from shared components; no official source copied.

- The local PatchTST backbone is composed with TSFLab's monotone quantile head; the cited paper's point-forecast results do not validate this probabilistic composition.

Cite: Nie et al., "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers", ICLR 2023 (arXiv:2211.14730).
