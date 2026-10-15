# PatchMLP — reference

## Paper

- **Title**: Unlocking the Power of Patch: Patch-Based MLP for Long-Term Time Series Forecasting
- **Venue**: AAAI 2025 (arXiv 2405.13575, 2024-05)
- **Abstract (shortened)**: Attributes much of the success of Transformer LTSF models to patching rather than self-attention, and shows that simple linear layers with patching can outperform them. Argues that cross-variable interaction is valuable but has been misapplied. PatchMLP uses moving averages to split smooth components from noisy residuals, mixes channels on the smooth part, and treats the noisy residual channel-independently.

## Implementation mapping

- Four-scale patch embedding (`Emb`, `EmbLayer`) and latent decomposition `X_s = AvgPool(X), X_r = X - X_s` on the embedding.
- The noisy residual branch remains channel-independent; the smooth branch adds dot-product-style inter-variable mixing after its intra-variable temporal MLP.
- Both branches use residual normalization before the horizon projection.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/TangZ25,
  author    = {Peiwang Tang and Weitai Zhang},
  editor    = {Toby Walsh and Julie Shah and Zico Kolter},
  title     = {Unlocking the Power of Patch: Patch-Based {MLP} for Long-Term Time Series Forecasting},
  booktitle = {Thirty-Ninth {AAAI} Conference on Artificial Intelligence, {AAAI} 2025},
  pages     = {12640--12648},
  publisher = {{AAAI} Press},
  year      = {2025},
  doi       = {10.1609/AAAI.V39I12.33378}
}
```
