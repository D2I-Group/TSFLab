---
name: "CALF"
description: "Twin GPT-2 branches over channel tokens; LoRA temporal branch aligned to a frozen principal-word textual branch. Use for multivariate forecasting with limited data that can borrow a pretrained GPT-2; not for more than 1024 channels or runs without the GPT-2 weights."
---

# CALF

## Idea

- `CrossModalMatch` embeds each channel's normalized window with `Linear(seq_len, 768)` and one Transformer encoder layer (time tokens), then cross-attends them to the principal components of GPT-2's word table to form aligned text tokens (Eqs. 2-3).
- Both branches use the shared `gpt2_backbone` truncated to `gpt_layers`; the temporal branch adds LoRA to every QKV projection and trains layer norms and positions, the textual branch trains positions only.
- `Model.load_gpt2` copies the pinned GPT-2 weights into both branches and refits the principal word embeddings by exact PCA.
- `cross_modal_loss` adds layer-wise feature alignment (`feature_weight`, decay `gamma`) and output consistency (`output_weight`) to the criterion (Eqs. 4-6).
- `forward` runs only the temporal branch, with an input residual around GPT-2 and a shared horizon head.

## When to use

- Limited training data where a pretrained language-model trunk with parameter-efficient tuning (LoRA) can transfer.
- Channels are tokens, so the model learns cross-channel dependence; attention is causal, so channel order matters, and at most 1024 channels fit.
- Each channel's whole window becomes one token: no patch-level temporal structure.
- Needs the GPT-2 artifact (`tsf model artifacts CALF --fetch gpt2`); without it both branches start random, which is not CALF.

## Configure

- `enc_in`: number of channels, at most 1024.
- No other data-dependent parameter.

Other hyperparameters: preset defaults in `configs/models/CALF.toml`; tune generically.

## Differences

Local rewrite after inspecting the pinned official code (Apache-2.0); neither `transformers` nor `peft` is imported.

- Loss weights follow the official code (`feature_weight = 0.01`, `output_weight = 1.0`); the paper text swaps them.
- The supervised term is the configured criterion; `alignment_loss` selects the similarity (`l1` default).
- Exact SVD replaces scikit-learn PCA (same scores up to sign).
- One optimizer for all trainable parameters instead of the official second Adam.
- Point forecasting only; classification, imputation, anomaly heads are out of scope.

Full detail in `reference.md`.
