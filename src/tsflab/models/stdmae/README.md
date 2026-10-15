---
name: "STDMAE"
description: "Temporal and spatial masked-autoencoder encoders over a long history inject context into a Graph WaveNet's skip path; pre-training is not included. Use for spatio-temporal node forecasting with a graph and long history; not for non-graph data or reproducing the paper's pre-trained results."
---

# STDMAE

## Idea

- `DecoupledMaskedEncoder(spatial=False)` is the T-MAE encoder: non-overlapping patch embedding, 2-D sinusoidal position over (node, patch), and a `tst_transformer` encoder attending along the patch axis within each node.
- `DecoupledMaskedEncoder(spatial=True)` is the S-MAE encoder: the same tokens, attending across nodes within each patch.
- `Model.encode_long_history` takes each node's last-patch representation from both encoders; two MLPs map them to the skip width and add them to the Graph WaveNet skip path before the end convolutions.
- The backend is a dilated gated-convolution Graph WaveNet over forward, reverse and adaptive supports (`diffusion_conv`, `graph_utils`, `adaptive_node_embedding_adjacency`) reading the last `history_len` steps with value and time-of-day.
- Masked pre-training is a separate stage not in this module; `freeze_encoders=True` freezes the encoders for externally pre-trained weights.

## When to use

- Traffic-style sensor graphs where long history (official PEMS04: 864 steps) gives context beyond the short 12-step window the backend reads.
- Predefined adjacency is used when available (identity otherwise) together with a learned adaptive adjacency.
- Without pre-training the encoders train end to end from scratch, which is not the paper's protocol. Point forecasts only.

## Configure

- `enc_in` follows the node count; `num_nodes` and `adj_mx` come from the dataset graph.
- `patch_size` follows `seq_len`: `seq_len` (the long history) must be a multiple of `patch_size`.
- `history_len` follows `seq_len`: `history_len <= seq_len`; the backend reads the last `history_len` steps.
- `in_dim` selects 1-3 of value, time-of-day, day-of-week from the loader's features.
- Other hyperparameters: preset defaults in `configs/models/STDMAE.toml`; tune generically (`embed_dim` divisible by `num_heads`).

## Differences

Independent implementation; the official `Jimmy-7664/STD-MAE@29fba76` has no license (`NOASSERTION`) and is reference-only.

- Pre-training (masking, decoders, reconstruction loss) and the released checkpoints are omitted; by default encoders train end to end, so results are not comparable with the paper.
- Context MLPs map `embed_dim -> 512 -> skip_channels` (official hard-codes 96 and 256); frozen encoders stay in evaluation mode.
- `sincos_2d` is written from the `positional_encodings` documented layout and not compared numerically with the package.
- Windows start where a full long history exists (official zero-fills early windows); calendar features come from repository marks.

Full detail in `reference.md`.
