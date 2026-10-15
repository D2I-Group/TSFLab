---
name: "TimeXer"
description: "Transformer that forecasts target series from their own patches plus exogenous variates via a global-token cross-attention bridge. Use for targets with informative external covariates and calendar marks; not for very many channels in M mode or covariate-free univariate data."
---

# TimeXer

## Idea

- `EndogenousEmbedding` turns each target series into non-overlapping patch tokens plus one learnable global token.
- `ExogenousEmbedding` embeds every whole series as a variate token plus one calendar token built from the time marks.
- In `TimeXerLayer` patch tokens self-attend; only the global token cross-attends to the exogenous tokens, and the feed-forward layer carries it back to the patches.
- A flatten linear head maps all tokens to the horizon; `revin` wraps the model.

## When to use

- Designed for forecasting a target from external variables (`MS`/`S`: the last channel is the target, the others are exogenous).
- Calendar marks feed one token, so timestamps that carry signal (hour, weekday, month) can help.
- In `M` mode every channel is a target that cross-attends to all channels, so cost grows with channels squared; avoid on hundreds of channels.
- Little benefit when the covariates are unrelated to the target (weakly correlated channels).

## Configure

- `enc_in`: number of input channels; must equal the dataset's channel count, with the target placed last for `MS`/`S`.
- `patch_len`: patch count is `ceil(seq_len / patch_len)` and the tail is zero-padded; choose a divisor of `seq_len` to avoid padding.

Other hyperparameters: preset defaults in `configs/models/TimeXer.toml`; tune generically.

## Differences

- Clean-room rewrite from the paper (the reference repository has no license); no source was copied.
- `MS`/`S` designate the last channel as endogenous; `M` treats every channel as an endogenous target against the shared external context.
- The calendar token is the window mean of four normalized marks (month, day, weekday, hour); without marks a fixed positional summary is used.
