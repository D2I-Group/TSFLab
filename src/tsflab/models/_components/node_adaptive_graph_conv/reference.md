# node_adaptive_graph_conv — reference

## Origin and granularity

The operator is AGCRN's Node Adaptive Parameter Learning (NAPL) combined with its
Data Adaptive Graph Generation (DAGG): Bai et al., "Adaptive Graph Convolutional
Recurrent Network for Traffic Forecasting", NeurIPS 2020 (`agcrn`). HimNet (Dong
et al., "Heterogeneity-Informed Meta-Parameter Learning for Spatiotemporal Time
Series Forecasting", KDD 2024, `himnet`) applies the same filter to per-sample
meta embeddings built from node, time-of-day, day-of-week and horizon
embeddings. Extracted from `agcrn` (`NodeAdaptiveConvolution` plus its
`_adaptive_basis` helper) and `himnet` (`MetaGraphConvolution`), which had
identical parameter names, shapes, initialisation and equations and differed
only in whether the embeddings carry a batch axis. The cut is the filter only:
how the embeddings are produced (a static parameter table in `agcrn`, a projected
meta context in `himnet`), the recurrent gating around it (`graph_conv_gru`) and
the readouts stay with the models.

## Invariants and equivalence evidence

- Both forward paths are verbatim moves of the model-local code: the `[N, D]`
  path is AGCRN's `_adaptive_basis` plus `NodeAdaptiveConvolution.forward`, the
  `[B, N, D]` path is HimNet's `MetaGraphConvolution.forward`, with the same
  einsum strings and operation order; parameter names, shapes, registration order
  and initialisation are unchanged, so checkpoints load as before.
- At extraction, frozen pre-extraction copies of both classes were compared:
  identical state-dict keys (ordered) and values under a fixed seed, identical
  outputs, and identical gradients with respect to the banks, `x` and the
  embeddings, for `order` 1 to 3. The same check covered the explicit formula
  above, that the batched path on expanded shared embeddings agrees with the static
  path, that `order=1` is node-local, gradient flow, and the `ValueError` for other
  ranks. No stored tensors: equivalence was pinned against the frozen inline copies.
- The full `agcrn` and `himnet` models were also compared with frozen pre-refactor
  copies (keys, outputs, gradients). These frozen-copy tests passed in the full suite run of 2026-10-03 before the test suite was consolidated.

## Variants and options

- Embedding rank selects the path: shared `[N, D]` keeps the basis and generated
  weights at `[K, N, N]` and `[N, K, I, O]`; per-sample `[B, N, D]` materialises
  `[B, K, N, N]` and `[B, N, K, I, O]` (`B` times the memory).
- Not covered: a separate source/target embedding pair, a predefined or mixed
  support, top-k sparsification, a non-Chebyshev polynomial, or dropout.

## Related components

`adaptive_node_embedding_adjacency` (the graph it builds internally),
`graph_conv_gru` (the GRU gating it is plugged into in `agcrn` and `himnet`),
`graph_spectral` (Chebyshev bases of a fixed Laplacian),
`regularized_adaptive_graph_conv` (linear-time node-embedding graph convolution),
`diffusion_conv` (shared-weight diffusion over static supports).
