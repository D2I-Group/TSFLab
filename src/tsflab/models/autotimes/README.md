---
name: "AutoTimes"
description: "Segment tokens through a frozen GPT-2 with next-segment training and autoregressive rolling forecasts. Use for long-context forecasting that reuses a pretrained language-model trunk with few trainable weights; not for calendar-text conditioning or offline runs without the GPT-2 weights."
---

# AutoTimes

## Idea

- Each channel is instance-normalized (`revin`, no affine) and cut into `seq_len / token_len` segments; `segment_mlp` maps each segment to one 768-d GPT-2 token (Eq. 2-3).
- A frozen GPT-2 trunk (`gpt2_backbone`, pinned released weights) maps token `i` to the hidden state of token `i+1`; a second `segment_mlp` projects every position back to a segment (Eq. 6-7).
- `next_segment_loss` supervises all positions against the history shifted by one segment plus the first future segment (Eq. 8); only the two MLPs learn.
- `forward` rolls out: predict the next segment, drop the oldest, append, repeat `ceil(pred_len / token_len)` times (Eq. 9).

## When to use

- Long contexts made of many segments (official script: 672 = 7 x 96) and arbitrary horizons through rolling forecasts.
- Few trainable parameters on top of a frozen pretrained trunk, which helps when training data is limited.
- Channel-independent; no timestamp text embeddings, so calendar effects are not used.
- Needs the GPT-2 artifact (`tsf model artifacts AutoTimes --fetch gpt2`); runs refuse to start without it.

## Configure

- `enc_in`: number of channels.
- `token_len`: `seq_len` must be a multiple of it, and `seq_len / token_len` must fit the GPT-2 context.

Other hyperparameters: preset defaults in `configs/models/AutoTimes.toml`; tune generically.

## Differences

Read the paper and the pinned official code (MIT); nothing copied. Parameters copied from the official `AutoTimes_Gpt2.Model` reproduced its next-segment outputs, rolling test loop, and training MSE within 6e-7.

- Backbone is the official GPT-2 variant (the paper's default is LLaMA-7B).
- Timestamp position embeddings (Eq. 4-5) are not implemented; marks are ignored (the official GPT-2 script also runs without them).
- `mlp_hidden_layers = 1` is rejected (the official code silently treats it as 2); `llm_layers` truncates the trunk for cheap tests.
- The context length is the task `seq_len` at training and inference.

Full detail in `reference.md`.
