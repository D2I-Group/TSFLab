---
name: "PENGUIN"
description: "Channel-independent patch Transformer whose grouped multi-query attention carries a periodic ALiBi bias, one head group per cycle length. Use for long-horizon forecasting of series with known, stable periods; not for aperiodic data or tasks needing cross-channel interaction."
---

# PENGUIN

## Idea

- `PeriodicNestedGroupAttention` splits `n_heads` into one group per period; each group shares one key/value head (grouped multi-query attention) and keeps its own relative-position bias.
- `periodic_alibi_bias` gives head `k` of a group the bias `-m_k * tri(|i - j| mod P_S)` with `m_k = 2^(-8k/n)`; periods are given in time steps and converted to patch units by dividing by `stride`.
- Encoder layer: attention, residual and norm, ReLU/GELU feed-forward, residual and norm, causal masking over patch tokens, final BatchNorm; `use_rmsnorm` swaps LayerNorm for RMSNorm.
- Inputs are RevIN-normalized (`revin`), embedded per channel by replicate-padded patching (`embed.PatchEmbedding`), and read out by `flatten_forecast_head`.

## When to use

- Long-term forecasting of series with one or more known, stable cycle lengths (for example daily and weekly in hourly data): each period gets its own head group, so several periods can be modelled at once.
- Not for series without a clear period: the bias then reduces to plain ALiBi distance decay and the period machinery adds nothing.
- Channels share one encoder and never interact; prefer a channel-mixing model when cross-channel dependence carries the signal.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.
- `periods` follows the dataset period: cycle lengths in time steps, each a positive multiple of `stride`; `n_heads` must be a multiple of `len(periods)` (and divide `d_model`).
- `patch_len` follows `seq_len`: `seq_len` must be at least `patch_len`.

Other hyperparameters: preset defaults in `configs/models/PENGUIN.toml`; tune generically.

## Differences

Paper and official code (pinned `d9a2b84`; no license file, nothing copied) were consulted for structure only.

- Periods are in time steps and converted to patch units (`P_S = P / stride`, paper Sec. 4.2.2); the official code passes them as token counts, which matches only its scripted `patch_len = stride = 1`.
- The bias is always on; slopes equal the official ones only when the heads per group are a power of two.
- The official `use_rcf` cycle-removal option and attention-weight output are not implemented; `causal` is an added switch (default on).
- Defaults differ from the official ETTh1 script (see reference.md).
