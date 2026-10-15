# graph_conv_gru — reference

## Origin and granularity

Replacing a recurrent cell's matrix products with graph convolutions was
introduced by Seo et al., "Structured Sequence Modeling with Graph Convolutional
Recurrent Networks" (GCRN, ICONIP 2018, arXiv 1612.07659); DCRNN (Li et al.,
"Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
Forecasting", ICLR 2018) made the GRU form with these exact gate positions the
standard spatiotemporal cell. Extracted from nine model-local cells with the same
gate order, split, state layout and interpolation `u*h + (1-u)*c`:
`dcrnn` `DCGRUCell` (DCRNN), `agcrn` `AdaptiveGraphGRUCell` (Bai et al., AGCRN,
NeurIPS 2020), `gts` `GraphGRUCell` (Shang et al., GTS, ICLR 2021), `pm25gnn`
`GraphGRUCell` (Wang et al., PM2.5-GNN, SIGSPATIAL 2020), `megacrn`
`MetaGraphCell` (Jiang et al., MegaCRN, AAAI 2023), `dgcrn`
`DynamicGraphGRUCell` (Li et al., DGCRN, ACM TKDD 2023), `graphlassomtsf`
`GraphGRUCell` (Do, Hy and Nguyen, GraphLASSO, arXiv 2306.17090), `himnet`
`MetaGraphGRUCell` (Dong et al., HimNet, KDD 2024) and `st_ssdl` `ChebGRUCell`
(ST-SSDL, NeurIPS 2025). Kept model-local: every graph filter, its
initialisation (e.g. `graphlassomtsf`'s `bias_start` of 1 for the gates), the
graph construction, the layer stacks, encoder-decoder loops and readouts.

## Invariants and equivalence evidence

- The gating arithmetic is the same operation sequence each consumer had
  (`cat`, `sigmoid`, half split, `tanh`, `u*h + (1-u)*c`); `chunk(2)` and
  `split(H)` select the same views when the gate width is `2H`, and the
  `st_ssdl` names `z, r` are the reset/update halves at the same positions.
- State-dict keys are unchanged for every consumer: six cells subclass
  `GraphConvGRUCell` and already used the attribute names `gates`/`candidate`
  with the gate filter registered first; `megacrn`, `pm25gnn` (`gates`/`candidate`
  linear layers) and `st_ssdl` (`gate_conv`/`candidate_conv`) keep their own
  modules and call `graph_gru_step`. Filters are still constructed gate-first,
  so seeded initialisation is unchanged.
- Equivalence against frozen verbatim pre-extraction copies of all nine cells was
  checked at extraction: for each, identical ordered state-dict keys and
  seeded values, identical eval outputs, and identical gradients for every
  parameter, the input, the state and the graph arguments (several orders for
  `agcrn`, `dcrnn`, `graphlassomtsf`). The same checks covered the explicit
  equations, submodule registration and graph-argument forwarding, saturated-gate
  limits and gradient flow. The frozen copies were code, not stored tensors. The
  full `agcrn` and `himnet` models were additionally checked against frozen copies.
  These frozen-copy tests passed in the full suite run of 2026-10-03 before the test suite was consolidated.

## Variants and options

- No options: the filters are the variation point. Consumers today use
  bidirectional random-walk diffusion (`dcrnn`, `gts`), single-support diffusion
  with a hop-minor layout (`graphlassomtsf`), static plus dynamic supports with
  dropout (`dgcrn`), node-adaptive filters (`agcrn`, `himnet` via
  `node_adaptive_graph_conv`), polynomial features plus `nn.Linear`
  (`megacrn`, `pm25gnn`) and shared-weight Chebyshev filters over several
  supports (`st_ssdl`).
- Not covered: GRU variants with separate reset and update filters, a candidate
  that applies the reset after the filter (`r * G(h)`), LSTM cells, layer
  normalisation inside the cell, or zoneout.

## Related components

`node_adaptive_graph_conv` (the filter inside `agcrn` and `himnet` cells),
`diffusion_conv` (Graph WaveNet diffusion in a `[B, C, N, T]` layout),
`graph_utils` and `adj_norm` (supports for diffusion filters),
`adaptive_node_embedding_adjacency` (self-learned graphs),
`graph_spectral` (Chebyshev supports of a fixed Laplacian).
