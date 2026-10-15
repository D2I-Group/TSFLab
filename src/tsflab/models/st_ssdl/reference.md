# ST-SSDL — reference

## Differences in detail

- Kept: the Chebyshev graph-conv GRU encoder/decoder (`ChebGraphConv`, `ChebGRUCell`), with the official gate-split naming where `z` mixes into the candidate and `r` blends the previous state with the candidate.
- Kept: the prototype memory with attention retrieval and top-2 nearest-prototype lookup, rewritten as the reusable `deviation_memory.PrototypeMemory` (matching the official `construct_prototypes` / `query_prototypes`).
- Kept: the data-driven decoding graph (`graph_hypernet` + `softmax(relu(e @ e^T))`), matching the official `hypernet` + `support`.
- Auxiliary losses: the official `STSSDL.forward(x, x_cov, x_his, y_cov, ...)` takes a second, distinct historical window `x_his` (e.g. the same lookback one period earlier). It trains the memory with a triplet contrastive loss (`nn.TripletMarginLoss`, query toward its nearest prototype and away from the second-nearest) and a deviation loss (`F.l1_loss`) matching the current-vs-historical query distance (`latent_dis`) to the current-vs-historical nearest-prototype distance (`prototype_dis`).
- TSFLab's forward signature `forward(x_enc, x_mark_enc, x_dec, x_mark_dec)` carries one lookback window, and a new public input is disallowed by the model contract. The official forecast decode needs only the current window (`h_de = cat([h_t, v_t])`), so `forward` implements the forecasting path faithfully; `Model.auxiliary_losses(x_enc, x_mark_enc, x_enc_hist, x_mark_enc_hist=None, margin=0.5)` must be called by a custom training script that supplies `x_enc_hist`.
- The official adaptive decoding graph uses `h_aug = cat([h_t, v_t, h_a, v_a])`; here `graph_hypernet` sees `cat([h_t, v_t])`, which changes the fitted weight's input distribution but keeps the architectural shape and role.
- `num_nodes` / `adj_mx` are runner-injected, matching STID/AGCRN in this repository. Default adjacency normalization is `symadj` (one static support), matching the official `--adj_type` default, via `graph_utils.adj_to_supports` (Apache-2.0 BasicTS-derived math) instead of `utils.load_adj`.
- The official time-of-day/adaptive/node-embedding front-end (`use_STE`) is not implemented; released configs default `use_STE=False`, which this path matches.
- Curriculum learning (`use_curriculum_learning`, `compute_sampling_threshold`) needs `labels` and a `batches_seen` counter the forward contract does not provide; the local decoder behaves as `use_curriculum_learning=False`.
- The preset uses smoke-scale widths (`rnn_units=16`, `prototype_num=8`, `prototype_dim=8`, `cheb_k=2`, `rnn_layers=1`) versus the official per-dataset configs (`rnn_units=64-128`, `prototype_num=20`, `prototype_dim=64`).

## Paper

- Title: How Different from the Past? Spatio-Temporal Time Series Forecasting with Self-Supervised Deviation Learning (NeurIPS 2025, arXiv:2510.04908).
- Motivation: recent spatio-temporal methods often fail to account for dynamic deviations between current inputs and historical patterns, limiting robustness and adaptability.
- Method: the latent space is discretized into prototypes, and a deviation-aware objective (contrastive and deviation losses) distinguishes the current system status from historical patterns.
- Evaluation: six real-world spatio-temporal datasets; the module can be integrated into other backbones as a general enhancement.
