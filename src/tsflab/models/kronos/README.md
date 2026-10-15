---
name: "Kronos"
description: "Decoder-only Transformer over binary-spherical-quantized coarse/fine tokens of each multichannel record, decoded autoregressively, trained from scratch. Use for jointly tokenized correlated channels such as financial K-line records; not as a zero-shot pretrained Kronos (no checkpoint)."
---

# Kronos

## Idea

- `HierarchicalTokenizer` maps each multichannel record to a spherical latent and binarizes it with straight-through Binary Spherical Quantization (`code_bits`), split into equal coarse and fine halves.
- Coarse and fine bit halves are embedded separately and fused (`_embed_bits`, `fusion`) into one token per step.
- `CausalBlock`s form a decoder-only Transformer with an incremental key/value cache (`prefill`, `step`) that generates the horizon autoregressively.
- The fine prediction is conditioned on a differentiable expected coarse code (`fine_context`); `tokenizer_loss` (coarse and full reconstruction plus BSQ commitment) is added to the criterion by `training_objective`.

## When to use

- Designed for financial candlestick (K-line) records, where price and volume channels form one record per step and are tokenized together.
- Channels must be meaningful as a joint record; weakly related channels gain nothing from shared tokens.
- Autoregressive decoding makes long horizons slower than one-shot heads.
- Not a zero-shot foundation model here: it trains from scratch with a compact tokenizer, so the paper's pretraining benefits do not apply.

## Configure

- `enc_in`: number of data channels; each multichannel record is tokenized jointly.

Other hyperparameters: preset defaults in `configs/models/Kronos.toml`; tune generically.

## Differences

- Compact clean-room rewrite of Eqs. (2)-(8), checked against `model/kronos.py` and `model/module.py` at the pinned revision; no source copied.
- Tokenizer is trained jointly with the forecaster (criterion on the decoded forecast plus `tokenizer_loss`); the paper's separate tokenizer pretraining and coarse/fine next-token cross-entropy are not reproduced, since forecasts are decoded from expected bits rather than sampled tokens.
- Affine tokenizer instead of the paper's Transformer autoencoder; 8-bit default vocabulary instead of 20 bits.
- Trained from scratch: no 12-billion-record corpus or pretrained weights.
