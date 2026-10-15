---
name: "DGCRN"
description: "Graph-GRU encoder-decoder whose directed graphs are regenerated at every step by hypernetworks from the hidden state and mixed with the predefined road graph. Use for traffic-like node data whose spatial correlations change over time; not for data without node structure."
---

# DGCRN

## Idea

- `DynamicGraphGenerator` is a hypernetwork mapping the current hidden state plus node embeddings to forward and backward row-normalized graphs at every step.
- `DynamicGraphConvolution` mixes predefined forward/reverse transitions with those dynamic graphs over multiple hops inside the GRU gates.
- One cell encodes history and decodes autoregressively, feeding back its projection; a known time driver comes from marks.

## When to use

- Road or sensor networks where inter-node correlations shift during the day, so a static graph alone is not enough.
- Needs a predefined adjacency to anchor the dynamic graphs (identity otherwise).
- Short horizons decoded step by step; point output only.

## Configure

- `enc_in`: number of graph nodes N.
- `adj_mx`: the dataset's `[N, N]` adjacency (identity when absent).

Other hyperparameters: preset defaults in `configs/models/DGCRN.toml`; tune generically.

## Differences

- Clean-room implementation from the method description; the MIT official code is reference only.
- Reduced default dimensions; one recurrent cell serves encoder and decoder.
- Task-level curriculum, target teacher forcing, official preprocessing, and published-metric reference comparison are not reproduced.
