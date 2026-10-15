---
name: "SCFormer"
description: "Channel-token Transformer with triangular-masked temporal maps, fed a HiPPO-LegS summary of history beyond the look-back window. Use for multivariate data with correlated channels where long history before the look-back carries signal; not for short series or inputs without extra history."
---

# SCFormer

## Idea

- `legs_transition` / `legs_projection` implement the cumulative historical state (Sec. 3.1, Eq. 1): the bilinear-discretized HiPPO-LegS recurrence is linear in the input, so the state after `H` steps is a fixed `[H, hippo_order]` projection applied per channel.
- The input is split: the last `lookback` steps are the look-back window; the `seq_len - lookback + 1` steps ending at the first look-back step are the history summarized by HiPPO.
- Eq. 2: `Z = MLP(Concat([MLP(l), c]))` embeds the instance-normalized window and the history state into one `d_model` series per channel, plus a learnable per-channel embedding.
- `StructuredLinear` / `StructuredAttention` (Sec. 3.2, Eqs. 4-7): attention runs between channel tokens while every map along the embedded series is lower-triangular masked; `struct_mode = "conv"` swaps in the official stacked 1D convolutions (Sec. 3.3).
- `EncoderLayer`: residual attention, LayerNorm, structured GELU feed-forward, residual LayerNorm; a final LayerNorm and linear head give the forecast, de-normalized with window statistics (Eq. 15).

## When to use

- Long series where history far before the look-back matters: HiPPO compresses it into a fixed-size state at the cost of a long input (`seq_len = lookback + 2047` reproduces the official 2048-step history).
- Multivariate data with correlated channels: attention is over channel tokens (iTransformer-style).
- With `seq_len == lookback` the history degenerates to one step and the model loses its distinctive input. Point forecasts only.

## Configure

- `enc_in` follows the dataset channel count; it must equal it exactly.
- `lookback` follows `seq_len`: `lookback <= seq_len`; the remaining `seq_len - lookback` steps are history (official: 2048 history steps, 256 for PEMS).
- Other hyperparameters: preset defaults in `configs/models/SCFormer.toml`; tune generically (`hippo_order` 512 by default, official PEMS scripts use 64; `d_model` divisible by `n_heads`).

## Differences

Independent rewrite of arXiv 2505.02655 (Eqs. 1-16) using the official `ShiweiGuo1995/SCFormer@a6f5aec` (model `SiTransformer`, no license, `NOASSERTION`) as reference only. Main differences (all recorded in `card.toml` issues):

- One consistent lower-triangular mask for all structured maps (official mixes `tril` and `triu`, breaking the temporal constraint).
- Follows the code where it departs from the paper: no ReLU on structured maps, two-sided conv padding in `conv` mode, kernel 16 for the stacked q/k/v convolutions, a per-channel embedding, 16 heads.
- History is a fixed window inside the input, not a cumulative state from the series start; the per-channel embedding is a real learnable parameter.
- Instance normalization uses `instance_norm`, not `revin`: the official std is not detached while `revin.RevIN` detaches it.

Full detail in `reference.md`.
