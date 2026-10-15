---
name: "TimePerceiver"
description: "Perceiver-style encoder-decoder: all channels' patch tokens are compressed by cross-attention into a few latents, and learnable queries tied to forecast timestamps decode the horizon. Use for multivariate forecasting with cross-channel dependencies and many tokens; not for probabilistic output."
---

# TimePerceiver

## Idea

- All channels' patch tokens (with channel and position embeddings) form one set that is compressed by cross-attention into `num_latents` latent vectors and refined by latent Transformer blocks, so cost scales with the latent count rather than tokens squared.
- Learnable target queries plus projected calendar features of the forecast timestamps cross-attend to the latents (`decoder`).
- A linear head emits all channels per step, or one value per (step, channel) query with `query_share=False`; `revin` wraps the model.

## When to use

- Multivariate forecasting where temporal and cross-channel dependencies should be captured jointly, at a cost bounded by the latent bottleneck (useful with many channels or patches).
- Data with calendar effects: decoder marks (month, day, weekday, hour) shape the target queries; without marks, fixed position features of the horizon are used.
- Past-to-future forecasting only; not for the paper's interpolation/imputation tasks or probabilistic output.

## Configure

- `enc_in`: number of channels (sizes channel embeddings and the output head; `query_share=False` creates `pred_len * enc_in` queries).
- `patch_len`: ideally divides `seq_len` (otherwise the lookback is zero-padded at the end).

Other hyperparameters: preset defaults in `configs/models/TimePerceiver.toml`; tune generically.

## Differences

- Clean-room implementation from the paper's input-segment encoder, latent bottleneck, latent self-processing, and timestamp-query decoder; reference source code was not copied.
- The official repository provides only past-to-future forecasting; the paper's generalized interpolation/imputation training sampler is unreleased and not implemented.
