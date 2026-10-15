---
name: "ExtraTreesTS"
description: "Ensemble of soft decision trees with frozen random axis-aligned splits over the flattened window and learned leaf forecasts, trained by gradient. Use as a cheap nonlinear tree-style baseline on small data; not as a faithful Extra-Trees or for long, many-channel windows."
---

# ExtraTreesTS

## Idea

- Samples a feature axis and a threshold per split once from a seeded generator and freezes them (`fixed_split_weight`, `fixed_threshold` of `soft_tree`); only leaf forecasts are learned.
- Routes the flattened `seq_len * enc_in` window softly through each tree (`SoftDecisionTree`) and averages the `num_estimators` trees' forecasts.
- Wraps the ensemble in reversible instance normalization (`revin`); no decomposition or channel-specific weights.

## When to use

- A low-variance nonlinear baseline in the spirit of Extra-Trees (random splits trade a little bias for less variance), useful when training data is small.
- Splits read any lag of any channel, so it can pick up cross-channel thresholds, but with few shallow trees it captures little temporal structure.
- Each split sees one feature of a flattened window; very long or many-channel windows dilute the random splits.

## Configure

- `enc_in`: number of channels; split axes are drawn over `seq_len * enc_in` features.

Other hyperparameters: preset defaults in `configs/models/ExtraTreesTS.toml`; tune generically.

## Differences

Clean-room baseline; no external source inspected or copied.

- Feature axes and normalized thresholds are sampled once and frozen; only leaf forecasts are learned.
- Soft routing and gradient fitting replace the Extra-Trees induction algorithm (random cut-points, full-sample unpruned trees); scikit-learn is not reproduced.

Citation: Geurts, P., Ernst, D., Wehenkel, L. "Extremely Randomized Trees." Machine Learning 63(1), 3-42 (2006). doi:10.1007/s10994-006-6226-1.
