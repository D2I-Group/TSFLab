---
name: "FreDF"
description: "Frequency Dynamic Fusion: per-rFFT-bin complex transfer matrices on the horizon-padded series, fused with learned weights. Use for hourly multivariate forecasting where frequency components differ in importance; not the FreDF frequency-domain loss (arXiv 2402.02399)."
---

# FreDF

## Idea

- The history is zero-padded by the horizon; `FrequencyDynamicFusionBlock` multiplies each rFFT bin by its own complex `d_model x d_model` matrix (Eq. 6), a learned transfer function per Fourier component.
- Per-bin outputs return to the time domain and are summed with trainable weights `frequency_weight` (Eq. 14); this is evaluated as one weighted spectrum, with `decoupled_reference` keeping Algorithm 1's literal K-copy form.
- Residual blocks `x + dropout(block(x))`; the last `pred_len` steps go through a horizon MLP and a `d_model -> channels` projection.
- Inputs are instance-normalized with non-affine `revin` and embedded with the TSLib `DataEmbedding` using optional calendar marks.
- This is arXiv 2407.12415 (Zhang et al., ACM MM 2024), not the loss of Wang et al. (arXiv 2402.02399, ICLR 2025).

## When to use

- Designed for series where different frequency components need different, input-dependent treatment rather than one shared mapping.
- Channels are mixed through the `d_model` embedding; the calendar embedding uses hourly marks.

## Configure

- `enc_in`: number of channels.
- `freq`: calendar-feature frequency; only hourly (`"h"`) is supported. Without marks the temporal embedding term is dropped.

Other hyperparameters: preset defaults in `configs/models/FreDF.toml`; tune generically.

## Differences

Independent rewrite; `Zh-XY22/FreDF` at `43ba9576` (no license file, recorded `NOASSERTION`) was read only for structure clarification (`models/FreDF.py`, `run.py`), nothing copied.

- Fusion weights are logits with a differentiable softmax (official: in-place re-softmax outside autograd).
- `irfft` gets an explicit length (official default is wrong for odd `seq_len + pred_len`).
- Shared `revin` detaches the standard deviation too (official: only the mean).
- As officially, one block is reused by every layer, so `e_layers` only sets the number of passes. Details in `reference.md`.
