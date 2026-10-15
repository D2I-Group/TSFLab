# PRISM — reference

## Differences in detail

- Implementation: independent rewrite from Section 3.2, Fig. 1 and Appendix A.2 / Tables 8-10 of the paper after reading the pinned official code (`nerdslab/PRISM`, revision `a6342da97fb763c901c803c5fc3ebd40e6ae6788`; the repository has no license file, so no license is asserted): `src/arch/prism.py`, `src/arch/layers/filters.py`, `src/arch/layers/SEBlock.py`, `src/exp/exp_long_term_forecasting.py`, `run_prism.py`, and `scripts/*_quick.sh`. Nothing was copied or imported.
- Split: both halves extend by `ov = min(overlap, ...)` samples past the midpoint (so the paper's overlap `o` corresponds to `2 * overlap`), and an odd segment's right half is trimmed to the left half's length.
- Haar: the analysis mirror-pads odd approximations by one sample and each band is synthesized to full length on its own; `haar_bands` uses `hi = (odd - even) / sqrt 2` with reflect padding. The EMA fallback (`tau_i = 8 * 3**i`) is used when `K - 1 > floor(log2 L)`.
- Gradient: band statistics and the decomposition run without gradient, so a child level's decomposition does not backpropagate into its parent's router.
- Router: six statistics per band (mean, std, max amplitude, mean absolute first and second difference, max / std); softmax temperature `0.2`, MLP hidden width `32`, one MLP per node serving both halves; the left/right band weights pass the second softmax over the two halves.
- Heads use hidden width `128` (the `enc_hidden`/`dec_hidden` options and `dropout` of the official scripts are never used by the model). The squeeze-and-excitation gate pools each band forecast over the horizon, applies two bias-free `N x N` linear maps with ReLU and a sigmoid.
- Normalization: per-series mean and sample (unbiased) standard deviation floored at `1e-5`.
- Preset follows `scripts/*_quick.sh`: `num_components = 5`, `tree_depth = 2`, `overlap = 8`, context 336, Haar, MAE loss, Adam at `1e-4`.
- Paper versus official code: the paper describes stitching the last level's bands back to the context length with a linear cross-fade and forecasting from that `T_context x C x K` signal with `M x K` MLPs, and trains with MSE; the official forecast path instead feeds the routed bands of each level (2K bands at the root, 4K at the second level, at their segment lengths) to per-band heads, never uses the cross-faded reconstruction, and all official scripts train with an L1 loss. This implementation follows the official forecast path.
- Dead nodes: in the official tree the leaf nodes (depth 0) and, with `use_last_layer_only`, every node below the root are evaluated but cannot influence the forecast (the leaves' band entry is overwritten by the root's and only root-level heads exist), and the stitched time-domain output is discarded in forecast mode; these nodes are not built here, which removes their unused router parameters but leaves the forecast unchanged.
- Not implemented: the plotting and band-visualization code, the filter banks used only in ablations (DoG, binomial, FFT, MCD, learnable Gaussian), and the unused hybrid feature extractor.
- No shared component is reused: the cataloged `haar_dwt1d`/`wavelet` transforms replicate-pad odd lengths and return decimated coefficients, `orthogonal_dwt` is likewise decimated, and `revin` normalizes with `sqrt(var + eps)`.
- Checked behaviour: official level sizes and overlapping split, Haar bands against hand-computed even and odd (mirror-padded) cases, exact partition of the signal and band orthogonality, the EMA ladder and the Haar/EMA dispatch without gradient, the six band statistics, the double temperature softmax, the squeeze-and-excitation gate, the level/head layout for the default and root-only settings, child inputs as weighted band sums, the full forward pipeline against a manual composition, shift and scale equivariance through the normalization, gradient flow to every router and head, and the parameter schema. Reported benchmark numbers are not reproduction claims of this implementation.

## Paper

PRISM: A hierarchical multiscale approach for time series forecasting, arXiv preprint 2512.24898 (2025-12).

## Citation

```bibtex
@article{chen2025prism,
  title   = {PRISM: A hierarchical multiscale approach for time series forecasting},
  author  = {Chen, Zihao and Andre, Alexandre and Ma, Wenrui and Knight, Ian and Shuvaev, Sergey and Dyer, Eva},
  journal = {arXiv preprint arXiv:2512.24898},
  year    = {2025}
}
```
