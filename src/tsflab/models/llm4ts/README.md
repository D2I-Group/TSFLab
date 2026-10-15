---
name: "LLM4TS"
description: "GPT-2 aligned to patched series by next-patch training with LoRA, then LP-FT fine-tuned with a flatten head. Use for data-efficient forecasting with calendar timestamps when a pretrained LLM trunk is acceptable; not for offline setups without the pinned GPT-2 weights or tight compute."
---

# LLM4TS

## Idea

- `PatchEncoding`: per-channel patches (replicate-pad `stride` steps, unfold) are embedded by a circular kernel-3 convolution (`embed.TokenEmbedding`) plus learned month/day/weekday/hour(/minute) tables of each patch's first timestamp (Eqs. 3-6).
- The first `first_k_layers` GPT-2 blocks (`gpt2_backbone`, released `gpt2` weights) add their trainable positions; `c_attn` carries `LoRALinear` adapters (rank 8, alpha 64).
- Stage 1 `alignment_loss`: standardize the window plus its next `stride` values, predict every next patch with `align_head`, MSE against the series shifted by one stride (Eqs. 8-9).
- Stage 2 `forward`: RevIN, the same encoder and GPT-2, a bias-free flatten-linear `forecast_head`, RevIN denormalization (Eqs. 10-11).
- `pretrain` runs alignment then linear probing; the TSFLab trainer then fine-tunes, completing LP-FT (`set_stage`).

## When to use

- Designed for data-efficient forecasting: a pretrained LLM is aligned to the series so few training windows suffice.
- Calendar marks carry signal through the temporal encoding; without marks the temporal term is dropped.
- Channel-independent with shared RevIN, so cross-channel structure is not modelled.
- Needs the pinned GPT-2 artifact and a GPU-scale budget; not for tight compute.

## Configure

- `enc_in`: must equal the channel count (channels are folded into the batch axis).
- `freq`: `h` for hourly-or-coarser marks, `t` for minute data (adds a 15-minute-bin minute table).
- `patch_len` / `stride`: `patch_len` must not exceed `seq_len + stride`; patches = `(seq_len + stride - patch_len) // stride + 1`.

Other hyperparameters: preset defaults in `configs/models/LLM4TS.toml`; tune generically.

## Differences

- Required artifact: `openai-community/gpt2` safetensors (revision `607a30d`, SHA-256 pinned), fetched with `tsf model artifacts LLM4TS --fetch gpt2`; runs refuse to start without it.
- LoRA is local with the official `peft` settings (target `c_attn`, i.e. Q, K and V; the paper names Q and K); AdaLoRA is not implemented.
- Temporal tables index integer marks as the paper describes; the official pipeline truncates continuous marks to index 0.
- Stages run inside `pretrain`, not as separate experiments; trainable sets follow the official code (GPT-2 LayerNorms stay frozen in fine-tuning).
- Channels are flattened into the batch instead of sampled one per example (same computation).
