---
name: "PMDformer"
description: "Patch Transformer that decouples each patch into its mean level and a zero-mean shape, attends across variables only at the latest patch, and restores trend through the attention values. Use for long-term forecasting of correlated multivariate series; not for univariate or independent channels."
---

# PMDformer

## Idea

- `patch_mean_decouple` splits each patch into its mean and a zero-mean shape residual; only the residual is embedded (Eqs. 1-3).
- Proximal variable attention lets the last patch token attend across variables, so channels mix only through their most recent context (Eqs. 4-5).
- `TrendRestorationAttention` computes Q and K from shape tokens only and adds the patch means to the values, restoring level information (Eqs. 6-8).
- `projection` flattens all patch tokens to `pred_len` (Eq. 9); `revin` wraps the model.

## When to use

- Long-term forecasting where local shape and absolute level should be modelled separately: attention matches shapes without being biased by the mean level, and levels are added back afterwards.
- Multivariate data whose channels interact, with the most recent inter-variable relations mattering most (PVA); the paper also reports lower memory than earlier patch Transformers.
- Not for univariate series or channels that move independently: the cross-variable block adds nothing there.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.
- `patch_len` follows `seq_len`: when `seq_len` is not a multiple, the history is left-padded by replicating the first value.

Other hyperparameters: preset defaults in `configs/models/PMDformer.toml`; tune generically.

## Differences

Compact clean-room rewrite of Eqs. (1)-(9), checked against `model/PMDformer.py` of the official repository at the pinned revision (no license file; nothing copied).

- One PVA block and one parameter-shared TRA block; single-head trend restoration.
- Non-divisible histories are left-padded by replicating the first observation.
- The full paper training configuration, numerical reference comparison, and reported hyperparameter sweep are not claimed.

Cite: Hu, Wen, Duan, Dai, He, Wang, Wang, Zhang, Jiang, Xu, "PMDformer: Patch-Mean Decoupling Information Transformer for Long-term Forecasting", ICLR 2026, https://openreview.net/forum?id=rfJ41gK9Ct.
