---
name: "Aurora"
description: "Patch Transformer guided by distilled text/image tokens, decoding via prototype retrieval and flow integration. Use for forecasting with side-modality context (dense text or image embeddings); not as a pretrained zero-shot foundation model (no weights bundled) or for sampled probabilistic output."
---

# Aurora

## Idea

- Patches each channel (`patch_len`) into tokens for a pre-norm `TransformerEncoder`, wrapped by `revin`.
- Learnable-query attention distils text and image tokens (optional dense embeddings; otherwise a domain token and an FFT magnitude token) that guide temporal tokens through `text_guider` and `image_guider`, mixed by `guide_gate`.
- Future queries cross-attend the encoded history to form per-step conditions.
- `prototype_retriever` picks from a bank of sinusoidal-initialized prototypes; `flow_network` integrates a deterministic velocity field for `flow_steps` steps.

## When to use

- Domains where text or image side information carries domain knowledge that the series alone lacks (the paper's multimodal setting).
- Dense modality embeddings enter only through `forecast_with_context`; the standard `forward` path uses the domain and FFT fallback tokens, so ordinary runs train a unimodal patch Transformer.
- No pretraining corpus or weights: zero-shot cross-domain claims of the paper do not apply.
- Channel-independent; point output from a deterministic mean flow.

## Configure

- `enc_in`: number of channels.
- `patch_len`: clipped to `seq_len`; the history is right-padded to whole patches.

Other hyperparameters: preset defaults in `configs/models/Aurora.toml`; tune generically. `d_model` must be divisible by `num_heads`.

## Differences

Local rewrite of Eq. (1)-(25) after inspecting the official `aurora/modeling_aurora.py`, `aurora/prototype_retriever.py`, and `aurora/flow_loss.py` at the pinned revision; nothing copied (no license file). Equations map to temporal patching, spectral guidance, learnable-query modality distillation, guided temporal attention, future-condition decoding, prototype retrieval, and velocity flow. Not bundled: BERT, ViT, the pretraining corpus, pretrained weights, raw text/image tokenizers, stochastic sampling, and zero-shot claims. Optional dense modality embeddings replace raw encoders; the registered point output follows a deterministic mean flow.
