---
name: "embed"
description: "Time-Series-Library embeddings: circular-conv token, sinusoidal position, fixed/learned/linear calendar, patch, and inverted (variate-token). Use for vanilla encoder-decoder Transformers or replicate-padded patch tokens; not for raw six-column marks with embed_type other than timeF, or learned/rotary positions."
---

# embed

## What it does

A module of independent embedding layers that map raw values and optional time
marks to `d_model` tokens. The main composites:

- `DataEmbedding`: `dropout(TokenEmbedding(x) + TemporalEmbedding_or_TimeFeatureEmbedding(x_mark) + PositionalEmbedding(x))`; with `x_mark=None` the calendar term is dropped.
- `DataEmbedding_inverted`: each variate's whole window becomes one token, `Linear(time -> d_model)` over `x.permute(0, 2, 1)`; marks (if given) are concatenated as extra tokens.
- `DataEmbedding_wo_pos`: `DataEmbedding` without the position term.
- `PatchEmbedding`: replicate-pad the end, unfold into patches of `patch_len` with `stride`, project with a bias-free `Linear(patch_len, d_model)`, add sinusoidal positions.

## When to use

Use for the vanilla encoder-decoder transformer family that expects
Time-Series-Library embeddings, or for patch tokenization with replicate padding
and sinusoidal positions. Do not use when marks are the repository's raw
six-column timestamps with `embed_type != "timeF"` (year would be read as month),
when learned/rotary positions are needed, or when a model needs
`forecast_embedding`'s normalized calendar projection instead.

## Interface

All classes are `nn.Module`. Public symbols: `PositionalEmbedding`,
`TokenEmbedding`, `FixedEmbedding`, `TemporalEmbedding`, `TimeFeatureEmbedding`,
`DataEmbedding`, `DataEmbedding_inverted`, `DataEmbedding_wo_pos`, `PatchEmbedding`.

- `PositionalEmbedding(d_model, max_len=5000)`: fixed sin/cos table buffer `pe` `[1, max_len, d_model]`. `forward(x)` returns `pe[:, :x.size(1)]` (only `x.size(1)` is read; dtype follows the float32 buffer); time must be <= `max_len`. `d_model` must be even (an odd value raises a shape error when the cosine half is assigned); the same holds for `FixedEmbedding`.
- `TokenEmbedding(c_in, d_model)`: bias-free `Conv1d` kernel 3, circular padding 1, Kaiming init. `forward(x [B, L, c_in]) -> [B, L, d_model]`. Parameter `tokenConv.weight`.
- `FixedEmbedding(c_in, d_model)`: frozen sinusoidal `nn.Embedding` of `c_in` indices; `forward(long index tensor)` returns the detached lookup.
- `TemporalEmbedding(d_model, embed_type="fixed", freq="h")`: sums month/day/weekday/hour (and minute when `freq == "t"`) embeddings. `embed_type="fixed"` uses `FixedEmbedding`, any other value uses learned `nn.Embedding`. Table sizes: minute 4, hour 24, weekday 7, day 32, month 13. `forward(x [B, L, >=4 or 5])` casts to long and reads columns by fixed index: 0 month, 1 day, 2 weekday, 3 hour, 4 minute. This is the five-column Time-Series-Library layout without year; the repository's six-column raw marks (year first) are not directly compatible.
- `TimeFeatureEmbedding(d_model, embed_type="timeF", freq="h", input_dim=None)`: bias-free `Linear(d_inp, d_model)` with `d_inp = input_dim` or 6 for every `freq` (the `freq_map` is constant 6; `embed_type` and `freq` are otherwise unused). Parameter `embed.weight`.
- `DataEmbedding(c_in, d_model, embed_type="fixed", freq="h", dropout=0.1, time_feature_dim=None)`: `embed_type="timeF"` selects `TimeFeatureEmbedding` (continuous marks, last dim must equal the chosen width, 6 by default), anything else selects `TemporalEmbedding`. `forward(x [B, L, c_in], x_mark [B, L, m] | None) -> [B, L, d_model]`.
- `DataEmbedding_inverted(c_in, d_model, embed_type="fixed", freq="h", dropout=0.1)`: note `c_in` here is the **time length** (the Linear input), and `embed_type`/`freq` are ignored. `forward(x [B, L, C], x_mark [B, L, M] | None) -> [B, C (+ M), d_model]`; without marks `c_in = L`, with marks it is still `L`.
- `DataEmbedding_wo_pos(c_in, d_model, embed_type="fixed", freq="h", dropout=0.1)`: as `DataEmbedding` without the position term; has no `time_feature_dim` argument (timeF width fixed at 6). It still constructs the unused `position_embedding`, so its state dict still contains the `position_embedding.pe` buffer.
- `PatchEmbedding(d_model, patch_len, stride, padding, dropout)`: `forward(x [B, n_vars, L]) -> (tokens [B*n_vars, P, d_model], n_vars)` with `P = (L + padding - patch_len) // stride + 1`. Note the input is channel-first, unlike the other classes.
- Buffers `pe` (all positional users) and frozen `FixedEmbedding` weights appear in the state dict. Dropout is the only stochastic element; no stateful caches. Float inputs; marks for the `fixed` path are converted with `.long()`, so out-of-range indices raise `IndexError`.
