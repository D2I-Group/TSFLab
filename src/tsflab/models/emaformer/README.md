---
name: "EMAformer"
description: "iTransformer variate tokens armored with channel, cycle-phase and joint channel-phase embeddings, then an MLP head. Use for multivariate data with a known stable cycle and regular timestamps; not for irregular sampling or data without a dominant period."
---

# EMAformer

## Idea

- Each variate's normalized lookback becomes one token (Eq. 4), as in iTransformer.
- `armored_tokens` adds a learnable channel embedding (Eq. 5), a phase embedding at `t mod cycle` (Eq. 6) and a joint channel-phase embedding (Eq. 7) to every token (Eq. 8).
- `forecast_phase` gets the cycle position of the first forecast step from the raw calendar marks.
- A post-norm full-attention encoder over variate tokens is followed by a `d -> 2d -> 4d -> pred_len` GELU MLP on the encoder output plus the embedded tokens.

## When to use

- Designed for data with a stable cycle (e.g. daily on hourly data) where each channel's behaviour depends on the phase; the embeddings give variate tokens channel identity and phase.
- Needs calendar marks with a constant sampling interval of at least one minute; without marks the phase is 0 and the phase tables lose their meaning.
- Attention over variate tokens mixes channels; cost grows quadratically with channel count.

## Configure

- `enc_in`: number of channels (channel and joint tables have one row per channel).
- `cycle`: the dominant period in steps (official ETTh1 script: 24).

Other hyperparameters: preset defaults in `configs/models/EMAformer.toml`; tune generically.
The preset follows the official ETTh1 script: `d_model` and `d_ff` 256, 4 heads, 2 layers, dropout 0.3.

## Differences

Rewritten for TSFLab after reading `PlanckChang/EMAformer` at `b8f37c0c` (MIT); nothing copied. The shared embedding, attention and encoder layers match the official ones line for line, and the full forward pass was checked against a tensor-op rewrite of the official forecast.

- Phase index from calendar marks at the first forecast step (as in the code; the paper uses the last observed step); this is a constant per-dataset relabeling of the official `row index mod cycle` and stays aligned across data gaps.
- Time-feature tokens are not embedded (the official model drops them before the encoder).
- `output_attention` is not exposed; the preset keeps `run.py` defaults rather than per-dataset scripts. Details in `reference.md`.
