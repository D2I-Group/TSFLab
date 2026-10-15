---
name: "Informer"
description: "ProbSparse-attention Transformer with distilling convolutions and a one-shot generative decoder for long sequences. Use for long-lookback, long-horizon forecasting where full attention is too costly and timestamps help; not for weakly correlated many-channel data (values are embedded jointly)."
---

# Informer

## Idea

- `ProbAttention` (`self_attention_family`) scores queries by sparsity and attends only with the top queries, giving O(L log L) attention.
- `ConvLayer` distilling between encoder layers halves the sequence length (`distil`), so long inputs stay affordable.
- The decoder (`transformer_encdec`) takes the label window plus placeholder horizon in one pass and projects to `c_out`, with causal ProbSparse self-attention and full cross-attention.
- `DataEmbedding` (`embed`) embeds values (mixing channels) and time marks; there is no instance normalization or decomposition.

## When to use

- Designed for long sequence forecasting: long lookbacks and long horizons where quadratic attention and step-by-step decoding are too slow or memory-hungry.
- Time-mark embeddings let calendar structure (hour, weekday) enter the model.
- Avoid on strongly non-stationary data without external normalization: there is no RevIN or decomposition.
- Channels are mixed in one token embedding, so it is a poor fit for many weakly correlated channels.

## Configure

- `enc_in`: number of data channels; `dec_in` and `c_out` default to it.
- `freq`: time-feature code matching the sampling interval (`h`, `t`, `d`, ...) when `embed = "timeF"`.

Other hyperparameters: preset defaults in `configs/models/Informer.toml`; tune generically.

## Differences

- Paper-driven local implementation using shared attention and Transformer primitives; no upstream source was copied, and the paper's benchmark numbers are not claimed.
- Uses TSFLab's common decoder-input contract.
- `ProbAttention` returns its context time-first (`[B, L, heads, d_head]`) as in the original Informer; the pinned Time-Series-Library revision keeps heads-first and views it without a transpose, which TSFLab does not reproduce.
- Batch or head count 1 is supported (explicit `squeeze(-2)` instead of a bare `.squeeze()`).
