---
name: "MTLinear"
description: "Linear forecaster that clusters variates by training-split correlation and gives each group its own shared DLinear-family head, trained with error-magnitude loss weights. Use for multivariate data with correlated variate groups under tight compute; not for cross-channel interaction, covariates, or probabilistic output."
---

# MTLinear

## Idea

- Treats each group of correlated variates as one task: `Model.fit_clusters` groups variates by complete-linkage agglomeration on `1 - corr` of the training series, once before training (`ModelSpec.training_setup`), stored in the `group_of` buffer.
- `Model.forward` routes each group to its own `GroupHead` (`dlinear` backbone, or `channel_wise_linear` with last-value centering for NLinear or `revin` for RLinear); variates in a group share weights, groups do not.
- `Model.penalty_weights` and `penalized_loss` implement the gradient magnitude penalty `w_ij = (K_j H_i)^-a` as detached per-batch constants; `ModelSpec.training_objective` replaces the configured loss with it whenever the horizon or variate penalty is on.

## When to use

- Multivariate data whose variates form correlated groups with distinct dynamics: one head per group avoids both a single shared head and one head per channel.
- Tight compute or memory budgets: every head is a linear map over `seq_len`.
- Not when cross-channel interaction carries the signal (heads are univariate within a group), when covariates or calendar marks matter (ignored), or when quantiles are needed (point output only).

## Configure

- `enc_in`: must equal the dataset's channel count.
- `cluster_dist`: `1 - corr` threshold for the grouping, which is fitted on the training split; merges happen strictly below it, so its effect depends on the dataset's correlation structure.
- `max_clusters`: cap on the number of groups; the effective count is `min(max_clusters, enc_in)`.

Other hyperparameters: preset defaults in `configs/models/MTLinear.toml`; tune generically.

## Differences

- Independent rewrite from Secs. 4.2-4.4 (Eqs. 7-8); the official repository has no license file and was read only to resolve omissions.
- `cluster_dist` is always a distance threshold (the official code treats an integer as a cluster count), and the grouping is re-cut to at most `max_clusters` groups.
- The grouping is fitted once from the scaled training series, not at construction; the official per-group early stopping is replaced by the runner's single rule.
- `kernel_size` is a parameter (official 25); per-dataset official settings are not preset. Full list in reference.md.
