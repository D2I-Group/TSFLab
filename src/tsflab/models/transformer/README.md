---
name: "Transformer"
description: "Vanilla encoder-decoder Transformer with full attention over channel-mixed time-step tokens, a causal decoder, and one-shot horizon output. Use as an attention baseline on multivariate data with calendar marks; not for long lookbacks or many channels where quadratic cost and mixed tokens hurt."
---

# Transformer

## Idea

- `DataEmbedding` embeds values and calendar marks per time step with a linear map over channels, so all channels share one token.
- `Encoder` stacks full self-attention (`FullAttention` in `AttentionLayer`) with feed-forward residual blocks.
- `Decoder` applies causal self-attention, encoder-decoder cross-attention, and a linear projection to the channels.
- The decoder input (label window plus placeholder horizon) returns the last `pred_len` steps in one pass.

## When to use

- A reference baseline for attention-based forecasting.
- Tokens mix all channels at each step, which assumes correlated channels; on weakly correlated channels channel-independent models usually fit better.
- Calendar marks enter through the time-feature embedding, so timestamp signal can be used.
- Full attention is quadratic in `seq_len`; avoid very long lookbacks.

## Configure

- `enc_in`: must equal the dataset channel count; `dec_in` and `c_out` follow it when unset.
- `freq`: must match the dataset sampling frequency (`h`, `t`, `d`, ...) so the `timeF` mark embedding has the right width.

Other hyperparameters: preset defaults in `configs/models/Transformer.toml`; tune generically.

## Differences

- Assembled from the paper's scaled dot-product attention, encoder/decoder residual blocks, causal decoder mask, cross-attention, and position-wise feed-forward layers using shared components.
- The time-series embedding and the one-shot forecast boundary are TSFLab integration choices, not part of the paper.
- Time-Series-Library is a reference only; no source was copied. Published benchmark numbers are not reproduced.
