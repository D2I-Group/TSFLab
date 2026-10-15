---
name: "TSRAG"
description: "Retrieval-augmented forecaster: top-k nearest context windows from a knowledge base, their futures fused with the query by an Adaptive Retrieval Mixer. Use for shifting dynamics when a context/future knowledge base is available; not for runs without one or for many channels."
---

# TSRAG

## Idea

- `retrieve` embeds the query and candidate contexts by parameter-free adaptive pooling and takes the top-k nearest by Euclidean distance; `retrieved_projector` embeds their future windows.
- `AdaptiveRetrievalMixer` (ARM) runs self-attention plus an FFN over the query and retrieved items, softmax-weights them, and adds the result to the query as a skip.
- `forecast_with_retrieval` takes an external knowledge base (`retrieval_contexts`, `retrieval_futures`); `forward` builds a fallback base of `memory_size` rolled copies of the input with interpolated futures.
- The query embedding flattens all channels of the `revin`-normalized window; a linear layer projects to the horizon.

## When to use

- The paper targets non-stationary dynamics and distribution shift, where retrieved similar segments supply context the input alone lacks.
- The benefit needs a real knowledge base passed to `forecast_with_retrieval`; through the common `forward` the retrieved items come from the input itself, so it behaves like a flatten-MLP.
- The input and output layers scale with `seq_len * enc_in` and `pred_len * enc_in`; avoid hundreds of channels.
- No TSFM backbone, so the paper's zero-shot setting is not available.

## Configure

- `enc_in`: must equal the dataset channel count; knowledge-base contexts must have shape `(N, seq_len, enc_in)` and futures `(N, pred_len, enc_in)`.
- `top_k`: must not exceed `memory_size` (the fallback base size).

Other hyperparameters: preset defaults in `configs/models/TSRAG.toml`; tune generically.

## Differences

- No third-party TSFM checkpoint, Chronos retrieval encoder, FAISS index, or pre-built multi-domain knowledge base.
- Retrieval descriptors use parameter-free adaptive pooling instead of a pretrained encoder; only the ARM is learned.
- `forward` uses a deterministic history-derived fallback so the standalone forecasting contract stays runnable.
- Implements paper Eqs. (1)-(12); `TS-RAG/retrieve.py` and `TS-RAG/models/ChronosBolt.py` were inspected at the pinned revision, no source copied.
