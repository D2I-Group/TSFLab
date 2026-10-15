---
name: "KNNForecasterTS"
description: "Soft nearest-reference baseline: distance-kernel weights over learned reference windows blend learned future continuations. Use as a small retrieval-style baseline for recurring window patterns; not for data with level shifts or trends beyond the training range (no normalization, outputs are blends of learned futures)."
---

# KNNForecasterTS

## Idea

- Holds `num_prototypes` learned reference windows (`reference_windows`) and matching learned future continuations (`reference_futures`) as parameters.
- `neighbor_weights` is a softmax of negative scaled mean squared distance between the whole input window (all channels) and every reference (`kernel_gamma`).
- The forecast is the weighted sum of reference futures; it is not a hard KNN over a stored training set.

## When to use

- Recurring window shapes that a few prototypes can cover; a cheap baseline for short training data or tight compute.
- The forecast is a convex blend of learned futures, so it cannot extrapolate levels or trends outside what the prototypes hold.
- No instance normalization: distances are taken on the scaled input, so level or scale shifts between train and test hurt.
- Distances span all channels jointly, so many channels dilute the match.

## Configure

- `enc_in`: number of data channels; reference windows and futures span all channels.

Other hyperparameters: preset defaults in `configs/models/KNNForecasterTS.toml`; tune generically.

## Differences

- Independent differentiable adaptation of the nearest-neighbor idea; no external implementation inspected or copied.
- Not a hard KNN over a stored training set: references and continuations are learned parameters and all contribute through a soft (RBF-style) kernel, so it trains end-to-end with the standard trainer on any device.
- The cited paper (Cover and Hart, "Nearest Neighbor Pattern Classification", IEEE Trans. Inf. Theory 13(1):21-27, 1967, doi:10.1109/TIT.1967.1053964) is conceptual background, not an equivalence claim.
