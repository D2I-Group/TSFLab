---
name: "PPGF"
description: "Classification-guided forecaster: bins each channel's training values into equal-count groups, predicts the group with a confidence-calibrated classifier and adds a learned within-group offset. Use for point forecasts whose value range is stable from train to test; not for level-shifting or trending data."
---

# PPGF

## Idea

- `group_edges` (Eqs. 1-2) fits `K` equal-count value groups per channel on the scaled training series; `assign_groups` turns a target into a group label and a relative offset `(y - left) / (right - left)`.
- `Model.extract` (Sec. III-B), per channel: `Conv1d(1 -> d, kernel 2)`, sinusoidal positions, one self-attention layer, and a variable-selection network of gated residual networks over positions, producing a general feature `g`.
- `Model.heads` (Sec. III-C): `h = FC1(g)`, an auxiliary TCP classifier, a confidence `c_hat` per horizon step trained to match the true-class probability, the final classifier on `h * c_hat` (Eq. 6), and a relative head `ReLU(FC3(g))` with one offset per group (Eq. 8).
- `Model.predict` (Eq. 10): most probable group, plus its offset times the group width, from the group's lower edge.
- `pattern_loss` (Eq. 11, `lambda = (1, 1, 5)`) replaces the configured loss: confidence MSE plus TCP cross-entropy, calibrated-classifier cross-entropy, and per-group offset regression.

## When to use

- Series whose future values stay within the range seen in training: the forecast is placed inside value groups fitted on the training split.
- Channels are forecast independently with shared weights, so it suits weakly correlated channels.
- Not for level shifts, trends, or regime changes that leave the training range: values outside it fall in the first or last group with an offset outside [0, 1]. Offsets are only bounded below, so a forecast can exceed its group's upper edge. Output is a point forecast.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels (one set of group edges per channel).
- `num_groups` edges follow the training split: `training_setup` fits the `K` equal-count edges on the scaled training series automatically (standard-normal quantiles clipped to [-3, 3] before that).
- `conv_kernel` follows `seq_len`: `seq_len` must exceed it; the variable-selection positions follow `seq_len`.

Other hyperparameters: preset defaults in `configs/models/PPGF.toml`; tune generically.

## Differences

Independent rewrite after reading the pinned official code (`syrGitHub/PPGF`, `a478f95`, no license file); nothing copied. The official code cannot run as published (missing module, missing data files) and selects checkpoints on the test split.

- Every channel is forecast independently and every horizon step has its own classifier, confidence and offset block; the code predicts one scalar of one channel.
- Group edges are fitted on the whole scaled training series of each channel (the paper's `A`), not on training targets.
- The TCP target is detached; loss weights are the paper's `(1, 1, 5)`; training uses the catalog optimizer and schedule.
- The unused `AR`, focal-loss and two-step variants are not implemented.
