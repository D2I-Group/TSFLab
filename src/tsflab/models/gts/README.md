---
name: "GTS"
description: "Jointly learns a discrete node graph via Gumbel-Softmax edge sampling and a bidirectional diffusion graph-GRU seq2seq. Use for multiple related series (sensor networks) whose graph is unknown or only approximate; not for independent channels or long horizons."
---

# GTS

## Idea

- `DiscreteGraphDiscovery` encodes each node's history, classifies every directed edge, and samples edges with straight-through Gumbel-Softmax in training (probabilities in evaluation), so the graph is learned end-to-end with the forecaster.
- A supplied adjacency is only a weak edge-logit prior and the target of `graph_prior_loss`; the training objective adds that BCE (weight 1) to the configured criterion only when an adjacency is supplied.
- `LearnedDiffusion` does bidirectional polynomial (Chebyshev-style) propagation on the sampled graph inside `GraphGRUCell` (shared `graph_conv_gru` gating), stacked as encoder and decoder.
- The decoder generates the horizon autoregressively from a zero start, one value per node per step.
- The paper casts structure learning as optimizing mean performance over a parameterized graph distribution, reported simpler and more efficient than bilevel graph learning.

## When to use

- Designed for multiple time series whose pairwise relations help forecasting but whose graph is unknown (or only approximately known); traffic sensor networks are the typical case.
- Mixes nodes through the learned graph; brings nothing when channels move independently.
- Edge classification is quadratic in the node count, and autoregressive GRU decoding suits short horizons.
- Point output only.

## Configure

- `enc_in`: the dataset's node count (runner-injected `num_nodes` takes precedence).
- `adj_mx`: optional known graph, injected by the runner; only a prior, the graph is still learned.

Other hyperparameters: preset defaults in `configs/models/GTS.toml`; tune generically.

## Differences

- Local implementation against TSFLab contracts; BasicTS (pinned revision) is a cited reference.
- Graph features use the current input window instead of a separate full-training-series feature file.
- Evaluation uses edge probabilities instead of random samples.
- The graph-prior BCE uses the supplied adjacency, not the official k-nearest-neighbour graph from the training series, and is active only when `adj_mx` is given.
- Encoder marks enter through `input_dim` (default 3: value, time of day, day of week); no future target is consumed. The official data pipeline, training schedule and published metrics are not reproduced.
- Citation: Shang, C., Chen, J., Bi, J. "Discrete Graph Structure Learning for Forecasting Multiple Time Series." ICLR 2021.
