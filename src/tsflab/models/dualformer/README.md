---
name: "Dualformer"
description: "Time- and frequency-domain Transformer branches fed depth-specific frequency bands, fused by a harmonic-energy periodicity gate. Use for long-term forecasting of heterogeneous or weakly periodic series where high-frequency detail matters; not for tasks needing per-channel independence or probabilistic output."
---

# Dualformer

## Idea

- `frequency_band_sampler` gives each encoder depth its own contiguous frequency band: shallow layers keep high-frequency detail, deep layers low-frequency trend (countering the Transformer low-pass effect).
- A time branch uses `FullAttention` encoder layers; a frequency branch uses a local `AutoCorrelationAttention` (FFT top-lag aggregation) in the same `transformer_encdec` layers.
- Layers are truly stacked on the running state (a fix over the pinned official code).
- `harmonic_energy_gate` weights the branches by periodicity evidence; the last fused step is projected to the horizon. `revin` and `forecast_embedding` (values plus calendar marks) frame the model.

## When to use

- Designed for long-term forecasting where deep Transformers lose high-frequency information; the paper reports its largest gains on heterogeneous or weakly periodic data.
- The gate adapts between time and frequency branches, so it is a hedge when periodicity strength varies across datasets.
- Channels are embedded jointly per step (token embedding), so it does not keep channels independent.
- Uses calendar marks when present (zero-filled otherwise).

## Configure

- `enc_in`, `c_out`: number of channels; must be equal.

Other hyperparameters: preset defaults in `configs/models/Dualformer.toml`; tune generically.

## Differences

Clean-room rewrite; `Akira-221/Dualformer` at `ebd4ccf8` (no license file, recorded `NOASSERTION`) was read only to resolve equation ambiguities.

- Depth-wise chaining fix: the official code recomputes each layer from the original embedding, so only the last layer counts; here all `e_layers` stack.
- Band-pass fix: unselected bins are zero-padded back to their true positions before the inverse FFT.
- Bands tile from shared integer edges `floor(n*k/L)` (the paper gives real-valued edges).
- Embedding is the shared `forecast_embedding`, not the official `DataEmbedding_wo_pos`.
- Only the training-style top-lag autocorrelation path is implemented (the official CUDA-only inference path is not).
- No checkpoint or metric comparison against the official recipe. Details in `reference.md`.
