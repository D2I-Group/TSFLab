---
name: "REPNet"
description: "Modular MLP: multi-scale dilated patch extractors with time-informed patches, residual mixer memory blocks, and summed per-extractor heads. Use for multivariate forecasting where multi-scale local patterns and calendar position matter; not without per-task configuration search."
---

# REPNet

## Idea

- `Representation` (Sec. 2.2): each patch extractor `(cover, dilation, stride)` takes windows of `cover` steps every `stride` steps, keeping every `dilation`-th sample; `ValueEncoder` embeds them with one of the Appendix C encoders (`linear` = Linear-GELU-Linear by default; `cnn_*` use per-variable grouped convolutions).
- `TimeEncoding`: selected time embeddings (`posEmb` fixed sin/cos table, `timeF` linear map of continuous calendar features, `tempEmb` learned calendar tables) are summed and standardized per sample; each extractor embeds the undilated time window and appends it, shared over variables, as the second half of the patch (time-informed patch).
- `MemoryBlock` (Sec. 2.3, Fig. 1): pre-LayerNorm residual blocks of a Linear over the patch axis with ReLU, optional `PatchAttention` across patches within each variable, and a GLU then a feature-axis Linear (per variable, or across variables with `joint_feature_mixing`); stacked `memory_layers` times.
- `Projection` (Sec. 2.4): the memory splits back into the K extractor groups; each optionally passes `lstm_layers` LSTM layers, is flattened per variable and mapped to the horizon by a bias-free Linear; the K forecasts are summed and RevIN-denormalized.

## When to use

- Multivariate data with local patterns at several scales: each extractor sees a different cover, dilation and stride.
- Data whose timestamps carry signal: `timeF` and `tempEmb` embed calendar marks into every patch (`posEmb`, the default, needs no marks).
- Strongly configuration-dependent: the paper reports the best of a per-task random search, and only three setups are released.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.
- `patch_extractors` follows `seq_len`: every `[cover, dilation, stride]` needs `cover <= seq_len`.
- `freq` follows the sampling frequency: calendar frequency of the marks for `timeF`/`tempEmb` (`"h"` hourly; `"t"` adds a quarter-hour table); other frequencies need marks already in Time-Series-Library `timeF` width.

Other hyperparameters: preset defaults in `configs/models/REPNet.toml`; tune generically.

## Differences

Independent rewrite from arXiv 2507.05891 (v1, with Appendix C) after reading `RobertLeppich/REP-Net` at `df3b2983` (no license file; nothing copied). Defaults reproduce the released ETTh1-96 setup.

- Attention, `tempEmb`, and the time-embedding statistics follow the paper's intent rather than the buggy official code (axis mix-up, constant calendar indices, batch-dependent statistics).
- Dense softmax attention (paper: sparse), ReLU after the positional linear (paper: GeLU), GLU before the feature linear (code order).
- Huber loss with unreported `delta` is not reproduced; runs use MSE. The future-mark embedding is unused and omitted.
