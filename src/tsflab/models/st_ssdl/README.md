---
name: "ST-SSDL"
description: "Chebyshev graph-conv GRU encoder-decoder with a learnable prototype memory that retrieves an expected-pattern state and conditions an adaptive decoding graph. Use for spatio-temporal node forecasting on a sensor graph with calendar covariates; not for non-graph multivariate data."
---

# ST-SSDL

## Idea

- Spatio-temporal methods often miss dynamic deviations between current inputs and historical patterns; ST-SSDL discretizes the latent space into prototypes to tell the current state apart from the past.
- Encodes the lookback with a Chebyshev graph-convolutional GRU (`ChebGRUCell`, shared `graph_conv_gru.graph_gru_step` gating) over static adjacency supports from `graph_utils`.
- Queries a learnable prototype bank (`PrototypeMemory` from `deviation_memory`) with the last hidden state to retrieve a soft expected-pattern vector appended to the decoder state.
- Builds a data-driven decoding graph from the prototype-augmented state and decodes the horizon autoregressively with a second `ChebGRUCell`, feeding future calendar covariates.
- The paper's contrastive and deviation losses live in `Model.auxiliary_losses`, which needs a historical reference window and is not called by `forward` or the runner; training uses the default loss.

## When to use

- Traffic and similar sensor networks: nodes with a known adjacency, short horizons decoded step by step, and time-of-day style covariates.
- Graph support is required for the static branch; without an adjacency the static support is the identity.
- In TSFLab the self-supervised deviation objective is absent, so the model behaves as a prototype-augmented graph GRU. Point forecasts only.

## Configure

- `enc_in` follows the node count (`num_nodes` defaults to it); `num_nodes` and `adj_mx` are injected by the runner from the dataset graph.
- `input_dim` follows the loader's input features: the value channel plus the calendar covariate channels driving the decoder (`input_dim - 1`).
- Other hyperparameters: preset defaults in `configs/models/ST-SSDL.toml` (smoke-scale widths; official configs use `rnn_units` 64-128, `prototype_num` 20, `prototype_dim` 64); tune generically.

## Differences

Local rewrite after reading arXiv:2510.04908 and `Jimmy-7664/ST-SSDL@a98331d` (MIT; `model_STSSDL/STSSDL.py`, `train_STSSDL.py`, `utils.py`). Encoder/decoder, prototype retrieval, and decoding-graph hypernet match the official operations.

- Auxiliary triplet and deviation losses need a second historical window `x_his`, which the TSFLab forward contract cannot carry; they are exposed as `Model.auxiliary_losses(..., x_enc_hist, ...)` and not used in training.
- The decoding graph is built from `cat([h_t, v_t])` only (official also uses the historical `h_a`, `v_a`).
- No `use_STE` front-end (official default off) and no curriculum learning; the decoder always feeds back its own prediction.

Full detail in `reference.md`.
