---
name: "Pyraformer"
description: "Pyramidal attention Transformer: strided convolutions build a multi-resolution tree of the lookback and each node attends only to its scale neighbours, children and parent. Use for long input windows under memory limits, with calendar marks; not when inputs need instance normalization against level shifts."
---

# Pyraformer

## Idea

- `CoarseScaleConstructor` builds coarser scales with strided `Conv1d`s (`window_size`, default 4 x 4) and concatenates all scales into one node sequence.
- `pyramid_neighbour_table` defines each node's attention set (PAM): `inner_size` neighbours at its scale plus its children and parent; `PyramidalAttention` attends only over that sparse set, so the signal path length is constant and cost linear in `L`.
- Inputs are a linear value embedding over all channels, normalized raw calendar marks through a learned projection, and sinusoidal positions; no instance normalization.
- The head concatenates the last finest node's ancestor chain across scales (`finest_ancestor_table`) and linearly emits `pred_len * enc_in` values.

## When to use

- Long lookbacks where full attention is too costly: the pyramid captures short- and long-range dependencies at linear time and memory.
- Data whose timestamps carry signal: raw calendar marks are embedded.
- Channels are mixed in the value embedding. There is no instance normalization, so level shifts between train and test hurt; prefer a RevIN model there.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.
- `window_size` follows `seq_len`: `seq_len` must be divisible by every successive branching factor (each at least 2), e.g. 96 with `[4, 4]`.

Other hyperparameters: preset defaults in `configs/models/Pyraformer.toml`; tune generically (`inner_size` must be odd).

## Differences

Clean-room rewrite from Eqs. (2)-(3) and Fig. 2 of the paper; the linked Time-Series-Library revision is kept only as a reference link and nothing was copied.

- Direct multi-horizon prediction (strategy 1), learned strided convolutions for CSCM, pre-normalized residual blocks, and dense gathers over a padded sparse-neighbour table: the pyramidal graph is preserved, the paper's optimized-kernel wall-clock complexity and published results are not claimed.
- Raw six-column marks `[year, month, day, weekday, hour, minute]` are normalized locally before a learned calendar projection.
- The earlier official reference candidate passed `dropout` positionally into an embedding argument and expected preprocessed time features; it was replaced, not declared equivalent.

Cite: Liu, Yu, Liao, Li, Lin, Liu, Dustdar, "Pyraformer: Low-Complexity Pyramidal Attention for Long-Range Time Series Modeling and Forecasting", ICLR 2022 (Oral); official code https://github.com/ant-research/Pyraformer.
