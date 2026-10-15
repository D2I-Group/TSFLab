---
name: "MoSSL"
description: "Spatial and modality attention with modality-spanning gated dilated convs, plus mixture-NLL and modality-contrast self-supervised losses. Use for multi-modality spatio-temporal forecasting (several measurements per node, e.g. bike/taxi in/outflow); not for plain multivariate series, graph priors, or long lookbacks."
---

# MoSSL

## Idea

- Inputs are flattened modality-major: `enc_in = num_modalities * L`, channel `m * L + l` is modality `m` at node `l`; `Model.to_tensor` lifts it to `[B, 1, M, L, T]` and a two-layer 1x1 ReLU projection gives `H_in`.
- `MoSTLayer` concatenates `[SA(H), MA(H), H]` (attention over nodes, Eq. 3, and over modalities, Eq. 2) and applies a gated `tanh * sigmoid` valid convolution of kernel `(M, 1, k)` at dilation `2^i` (Eq. 4); summed skips feed a ReLU-conv-ReLU-conv predictor (Eq. 12).
- `MultiModalityAugmentation` (Sec. 3.2) masks `(node, modality)` cells of `H_in` with probability proportional to `1 - relevance`, adds the MoST embedding `e_t + e_n + e_m`, and re-encodes with the shared encoder.
- `GlobalSSL` (Eqs. 7-9) scores the original representation by a Gaussian-mixture NLL predicted from the augmented one; `ModalitySSL` (Eqs. 10-11) contrasts own-modality and other-modality nodes with a bilinear form under binary cross-entropy.
- In training, `aux_loss = L_g + L_c` is added to the regression loss (Eq. 13); evaluation runs only the original view.

## When to use

- Spatio-temporal data with several modalities per node (the paper: NYC bike/taxi inflow and outflow, Beijing air quality), where node and modality interactions both carry signal.
- Short fixed input windows: the dilated stack consumes exactly `seq_len` steps (16 by default).
- Not for plain multivariate series without a modality structure (with one modality, `L_c` is zero), when a predefined graph or calendar marks should be used (ignored), or for long lookbacks.

## Configure

- `enc_in`: `num_modalities * nodes`, channels ordered modality-major (channel `m * L + l`).
- `num_modalities`: the dataset's modalities per node; must divide `enc_in`.
- `layers`, `kernel_size`: follow `seq_len`; `seq_len` must equal `1 + (kernel_size - 1)(2^layers - 1)` (16 for the default 4 layers, kernel 2).

Other hyperparameters: preset defaults in `configs/models/MoSSL.toml`; tune generically.

## Differences

- Independent rewrite from the paper (Secs. 3-4) and the official code at `2ca992ea` (no license; nothing copied). Where they disagree it follows the code (dilations 1, 2, 4, 8; two-layer input projection; averaged losses; see issues).
- Fixes official bugs: GSSL uses the Eq. 9 likelihood with summed log densities, MSSL negatives always come from another modality, and the MoST embeddings are trained parameters.
- The augmented view and SSL losses run only in training; `self_supervised = false` gives plain MSE training of the MoST Encoder.
- Marks and adjacency are not used (the method has none). No reference comparison was run.
