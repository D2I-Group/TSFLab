---
name: "segment_mlp"
description: "Segment-to-token or token-to-segment MLP over the last axis: one Linear or n >= 2 Linear layers with activation and dropout. Use for LLM-backbone forecasters that tokenize non-overlapping segments (AutoTimes, TALON); not for positional patch embeddings, history-to-horizon heads, or flatten heads."
---

# segment_mlp

## What it does

LLM-backbone forecasters that cut each channel into non-overlapping segments
need a map from a segment of `token_len` values to one LLM token, and from an
LLM hidden state back to a segment. `SegmentMLP(f_in, f_out, hidden_dim,
hidden_layers, dropout, activation)` is that map, applied over the last axis:

- `hidden_layers = 0`: `y = W x + b`;
- `hidden_layers = n >= 2`: `h_0 = x`, `h_i = dropout(act(W_i h_{i-1} + b_i))`
  for `i = 1..n-1` (widths `f_in -> hidden_dim -> ... -> hidden_dim`), then
  `y = W_n h_{n-1} + b_n` (`hidden_dim -> f_out`).

Segmenting, instance normalization, and the LLM itself stay with the caller.

## When to use

Use in forecasters that reuse a language-model backbone on non-overlapping
segments of each channel: embed a segment into the LLM token width, or project an
LLM hidden state back to a segment. Do not use for patch embeddings with
positional encodings (see `embed`), for channel-wise history-to-horizon
projections (see `channel_wise_linear`), or for flatten-style heads over patch
tokens (see `flatten_forecast_head`).

## Interface

- `SegmentMLP(f_in, f_out, hidden_dim, hidden_layers, dropout, activation)`:
  `nn.Module`; `forward(x [..., f_in]) -> [..., f_out]` (any leading axes).
  `hidden_layers == 1` (or negative) raises `ValueError("mlp_hidden_layers must
  be 0 (linear) or at least 2")`; an unknown `activation` raises `KeyError`
  (callers validate against `ACTIVATIONS` first). With `hidden_layers == 0`,
  `hidden_dim`, `dropout`, and `activation` are ignored.
- State: one `nn.Sequential` named `layers`; state-dict keys `layers.{3i}.weight`
  / `layers.{3i}.bias` for the hidden Linear layers (activation at `3i+1`,
  dropout at `3i+2`) and the last Linear at index `3(n-1)`; with
  `hidden_layers == 0` only `layers.0.*`. Default `nn.Linear` initialization;
  no buffers.
- `ACTIVATIONS`: `{"relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}`, the
  accepted activation names.
- Dropout is active in training mode only.
