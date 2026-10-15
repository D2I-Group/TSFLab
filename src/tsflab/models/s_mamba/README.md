---
name: "S_Mamba"
description: "Simple-Mamba: each variate's lookback becomes one token, a bidirectional Mamba scan over variate tokens models inter-variate correlation, and an FFN models temporal features. Use for multivariate data with correlated channels, including many channels; not for univariate series."
---

# S_Mamba

## Idea

- `InvertedTokenization` embeds each variate's whole lookback as one token via a linear layer, as in iTransformer.
- `SMambaLayer` runs forward and time-flipped `MambaBlock` scans (`mamba` component) over the variate-token sequence to extract inter-variate correlation.
- A feed-forward network (`temporal_ffn`) then learns each variate's temporal dependencies.
- `output_projection` maps each token to `pred_len`; inputs are standardized per window (`use_norm`).

## When to use

- Multivariate data whose channels are correlated: the scan runs across variates, with near-linear cost in their number (the paper's motivation versus quadratic Transformer attention).
- Large channel sets such as traffic or electricity, where full variate attention is costly.
- Not for univariate series (a one-token sequence); the shared pure-PyTorch scan is slower than the CUDA kernels.

## Configure

- `enc_in` follows the channel count: must equal the number of input variates.

Other hyperparameters: preset defaults in `configs/models/S_Mamba.toml`; tune generically.

## Differences

Clean-room implementation of the paper algorithm; the author repository (no license file) is reference-only and no code was copied.

- Token-axis layout, bidirectional scan, FFN, and projection are checked locally.
- Inputs are `[B, seq_len, variates]`, outputs `[B, pred_len, variates]`; marks are ignored.

Cite: Wang, Kong, Feng, Wang, Yang, Zhao, Wang, Zhang, "Is Mamba effective for time series forecasting?", Neurocomputing 619:129178, 2025, doi:10.1016/j.neucom.2024.129178 (arXiv:2403.11144).
