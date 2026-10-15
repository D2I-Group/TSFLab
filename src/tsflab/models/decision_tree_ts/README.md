---
name: "DecisionTreeTS"
description: "Single differentiable soft binary tree (sigmoid routing, learned leaf values) over the flattened multichannel lag window. Use as a simple tree baseline on small-channel multivariate data; not for many-channel data, CART-style interpretability, or probabilistic output."
---

# DecisionTreeTS

## Idea

- `soft_tree` (`SoftDecisionTree`) routes the input through depth `tree_depth` with learned soft splits, so the whole tree trains by gradient descent.
- Time and channels are flattened into one feature vector; the leaves output the full horizon for all channels.
- `revin` normalizes and denormalizes.

## When to use

- A sanity-check tree baseline for multivariate windows where joint lag/channel features matter.
- Input and leaf widths scale with `seq_len * enc_in` and `pred_len * enc_in`; not for many-channel data.
- Not a CART replacement: no hard splits to inspect; point output only.

## Configure

- `enc_in`: the dataset's channel count; inputs must be exactly `[B, seq_len, enc_in]`.

Other hyperparameters: preset defaults in `configs/models/DecisionTreeTS.toml`; tune generically.

## Differences

- Clean-room, end-to-end differentiable soft tree, not CART: no greedy impurity-reducing hard splits, no pruning, no scikit-learn reproduction.
- Classical CART (lagged features, axis-aligned splits minimising an impurity criterion, regularised by depth and leaf size) is conceptual background only: Breiman, Friedman, Olshen and Stone, *Classification and Regression Trees*, Wadsworth 1984 (ISBN 0-534-98053-8). No external source code was inspected or copied.
