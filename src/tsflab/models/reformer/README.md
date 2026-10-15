---
name: "Reformer"
description: "Efficient Transformer with shared-QK locality-sensitive-hashing attention and reversible residual blocks, run over history plus future placeholders with calendar marks. Use as a sparse-attention baseline for long sequences; not as a fast model in TSFLab (bucket loops are slow) or against level-shifted data."
---

# Reformer

## Idea

- `LSHSelfAttention` shares query and key projections, hashes them with random rotations over `n_hashes` rounds, sorts into buckets of `bucket_size`, and attends within each chunk and the previous one; duplicates across rounds get a log-multiplicity correction. This changes attention cost from O(L^2) to O(L log L).
- `ReversibleBlock` splits the width in half (`y1 = x1 + f(x2)`, `y2 = x2 + g(y1)`) and provides an `inverse`, so activations need not be stored per layer in the paper's training scheme.
- History and `pred_len` future placeholders (from `x_dec`) are embedded together by `TimeValueEmbedding` (value over all channels plus 6 calendar marks); the last `pred_len` positions are projected to `c_out`.
- The attention is implemented with Python loops over buckets: exact to the method but slow.

## When to use

- Long input sequences where full attention is the memory bottleneck, as a historical efficient-Transformer baseline.
- Data whose timestamps carry signal: six-column calendar marks are embedded with the values.
- Channels are mixed in the value embedding and there is no instance normalization, so level shifts between splits hurt; the local implementation does not deliver the paper's speed.

## Configure

- `enc_in` follows the channel count: must equal the number of input channels.
- `c_out` follows the channel count: number of output channels (defaults to `enc_in`).

Other hyperparameters: preset defaults in `configs/models/Reformer.toml`; tune generically (`d_model / 2` must be divisible by `n_heads`).

## Differences

Clean-room implementation from the paper; the linked Time-Series-Library wrapper is reference-only and no source was copied.

- Sparse candidate width, causal masking, duplicate correction, shared-QK hashing, and reversible inversion are checked structurally.
- Inputs are history plus future placeholders and optional six-column marks; outputs are `[B, pred_len, c_out]`.
- Training uses standard autograd; the paper's custom reversible-memory saving is not claimed.

Cite: Kitaev, Kaiser, Levskaya, "Reformer: The Efficient Transformer", ICLR 2020 (arXiv:2001.04451).
