---
name: "RandomForestTS"
description: "Differentiable random-forest baseline: an average of soft decision trees, each seeing a fixed random subset of the flattened lag window and emitting the whole forecast. Use as a tree-ensemble reference on small, short-train problems; not for long lookbacks with many channels or as a scikit-learn forest."
---

# RandomForestTS

## Idea

- `SoftDecisionTree` gives a differentiable tree of depth `tree_depth` whose leaves output the full `pred_len * enc_in` forecast.
- Each of `num_estimators` trees has a fixed random `split_mask` (`feature_fraction`) over the flattened `seq_len * enc_in` input, seeded by `random_seed`.
- The forecast is the mean over trees (random-forest averaging), trained by gradient descent rather than greedy splitting.
- `revin` normalizes the input; all channels enter each tree jointly.

## When to use

- A tree-ensemble baseline with small capacity, useful on short training sets and as a contrast to linear and deep models.
- All channels enter jointly, so cross-channel information is available to every tree.
- Not for many channels with long lookbacks (the flattened input grows as `seq_len * enc_in`, and every leaf emits `pred_len * enc_in` values); not equivalent to a classical random forest.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.

Other hyperparameters: preset defaults in `configs/models/RandomForestTS.toml`; tune generically.

## Differences

Clean-room baseline; the cited paper supplies the ensemble principle only, and no external source code was inspected or copied.

- Averages independently parameterized soft trees with deterministic random feature masks.
- No bootstrap of training rows, greedy hard splits, out-of-bag error, or scikit-learn reproduction.

Cite: Leo Breiman, "Random Forests", Machine Learning 45(1):5-32, 2001, doi:10.1023/A:1010933404324.
