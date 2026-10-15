---
name: "SiamTST"
description: "Channel-independent pre-RMSNorm, QK-normed patch Transformer, pre-trained by masked patch reconstruction plus Siamese shifted-window similarity, then a fine-tuned linear head. Use for self-supervised pre-training on the training split before forecasting; not for cross-channel dependencies or probabilistic output."
---

# SiamTST

## Idea

- `Backbone` (Eqs. 1-10): non-overlapping patches from the end of the window, `Linear(patch_size, d_model)` plus a learnable positional table, then `e_layers` pre-norm layers (RMSNorm, bias-free attention, bias-free GELU feed-forward of width `4 d_model`) and a final RMSNorm; channels are independent series with shared weights under `revin`.
- `Attention` applies RMSNorm to each head's queries and keys before scaled dot-product attention (QK-Norm, Eqs. 5-7).
- `pretraining_loss`: a random ratio in `[min_mask_ratio, max_mask_ratio]` zeroes whole patches, `pretrain_head` reconstructs them (masked MSE), and `1 - cosine` similarity ties the latents of shared patches of the window and a forward-shifted copy (same mask, no gradient); weights `1 - alpha` and `alpha`.
- `pretrain` runs that objective on the training split (AdamW, one-cycle), then freezes all but the forecasting head (`Flatten -> Dropout -> Linear`), which ordinary training fits; RevIN is inverted on the forecast.

## When to use

- When self-supervised pre-training on the target's own training split is worth an extra stage before a linear-probe forecast (the paper targets telco network KPIs).
- Channel-independent with shared weights: suits weakly correlated channels, not data whose signal lies in channel interactions.
- Pre-training costs `pretrain_epochs` extra passes; point forecasts only.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly.
- `patch_size` follows `seq_len`: `seq_len >= patch_size`.
- `stride` follows `pred_len`: pre-training needs `pred_len >= stride` and at least two patches (the Siamese view shifts into the horizon).
- Other hyperparameters: preset defaults in `configs/models/SiamTST.toml`; tune generically (`d_model` divisible by `n_heads`).

## Differences

Independent rewrite of Sec. 3 of arXiv 2407.02258 with the pre-training objective from `simenkristoff/SiamTST@f8069d5` (Apache-2.0); nothing copied. The official code cannot run at this revision (constructor, loss and `train.py` errors; see `card.toml` issues).

- `pre_norm` and `qk_norm` default to on (paper); the official flags default to off.
- Each Siamese view is instance-normalized with its own statistics; attention dropout only in training.
- The Siamese view is shifted by `min(patches // 2, pred_len // stride)` patches using training-horizon values (official: `seq_len // 2` steps later in a separate dataset).
- Pre-training uses only the run's training split (the paper also pre-trains across sectors) and keeps the final, not best-validation, state; missing-value imputation is not reproduced.

Full detail in `reference.md`.
