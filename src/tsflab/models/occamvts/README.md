---
name: "OccamVTS"
description: "Compact OccamVTS deployment student: a channel-independent patch Transformer cross-attending a small conv encoder over raw, spectrum and periodic channels, trained directly without the vision teacher. Use for lightweight forecasting of seasonal series; not for the paper's distilled few-shot or zero-shot protocol."
---

# OccamVTS

## Idea

- Implements only the inference-time student; the large vision teacher and the distillation losses are not part of this module.
- `temporal_encoder` runs a Transformer over overlapping patch tokens (`patch_len`, `stride`) of each channel.
- `visual_augmentation` stacks the raw series, its normalized FFT magnitude, and sine and cosine at `period`; `visual_encoder` convolves that stack.
- `cross_modal` attention lets temporal tokens query the visual features; pooled tokens go to a linear horizon head; `revin` wraps the model.

## When to use

- Tight compute or parameter budgets: the paper's point is that about 1% of a vision model's parameters suffices; the student is small (default `d_model = 32`, one layer).
- Series with a known dominant period, which feeds the periodic visual channels, and with spectral structure visible in the FFT magnitude.
- Weakly related channels: every channel is forecast independently with shared weights.
- Not when cross-channel interaction, exogenous inputs or calendar marks matter (marks are ignored), when quantiles are needed, or to reproduce the paper's teacher-distilled few-shot or zero-shot results.

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count.
- `period`: dominant seasonal period in steps for the sine/cosine channels; follows the sampling frequency (default 24 = daily cycle of hourly data).
- `patch_len`, `stride`: follow `seq_len`; `patch_len` is clipped to `seq_len` and `stride` to `patch_len`; token count is `1 + (seq_len - patch_len) // stride` (no end padding).

Other hyperparameters: preset defaults in `configs/models/OccamVTS.toml`; tune generically.

## Differences

- Clean-room implementation of the deployment student from Eqs. (1), (2), (8), (9) and (12): overlapping patch tokens fused with compact visual features from raw, FFT-magnitude and periodic channels. The unlicensed repository (`NOASSERTION`) is reference-only; its source was not inspected or copied.
- The pretrained vision teacher, pseudo-image resizing, pyramid feature alignment, and correlation/feature distillation objectives are training-only and not included.
- The preset is therefore a compact student trained directly for forecasting, not a reproduction of the paper's distilled weights, few-shot results, or zero-shot protocol.
