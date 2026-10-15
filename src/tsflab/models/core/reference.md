# CoRe — reference

## Paper

- **Title**: Correction-space Cross-variate Interaction for Test-time Adaptation in Time Series Forecasting
- **arXiv**: https://arxiv.org/abs/2609.34638 (v1, 2026-09-28)
- **Code**: https://github.com/yyddou/CoReTTA at `7e3861f24e3aaece9fffc048fd585b3d283835d8`; no LICENSE file, and the repository asserts no repository-wide license (it lists TAFAS and COSA third-party notices). Treated as read-only reference.

## Architecture map

1. Frozen base forecast `Y_base` (self-contained last-value fallback, or any external frozen forecast).
2. Base adapter: per-variate (or shared) linear map of `[Y_base_c || context]` scaled by `tanh(gate)` (variate-wise gate by default).
3. Shared correction anchor `alpha = mean_c Delta_c` and concatenation `[Delta_c || alpha]` (Eqs. 3-4).
4. Rank-`r` bottleneck `delta_c = W_up tanh(W_down z_c)`, shared across variates, `r = C` by default (Eq. 5).
5. Spectral descriptor `s = [SE, LBR, MBR, HBR]` of the input window and gate `g = tanh(W_g s + b_g)` per variate (Eqs. 7-8, 10-17).
6. Output `Y = Y_base + Delta + g * delta` (Eq. 6).

Sections 3.2-3.6 and Appendix B.1-B.3 of the paper were read, and `tta/core.py` (`SimpleOutputAdapter`, `_compute_spectral_context`) was inspected at the pinned revision to resolve initialization and tensor layout. Component decisions: `channel_wise_linear` reused (frozen last-value base and per-variate/shared base adapter); `spectral_descriptor` extracted as a new paper-neutral component (Eqs. 10-17); the SCR bottleneck, anchor, and gate wiring stay model-local.

## Differences in detail

- The streaming protocol is not run: delayed-label buffer, batch-wise adaptation loop, PAAS, adaptive
  learning-rate schedule, and weight-decay objective (Eq. 9) are outside the model. `adaptable_parameters()`
  exposes the adapted set; a frozen last-value base and a channel-mean history context are explicit
  self-contained fallbacks, whereas the official context is a buffer of past revealed target means.
- The official code computes one spectral descriptor per adaptation batch (power spectrum averaged over the
  batch); the paper defines it per input window, and this implementation computes it per sample (`[batch, 4]`).
- Band edges follow the paper's inclusive index ranges (Eqs. 13-15); the official half-open thirds
  (`[0,K//3)`, `[K//3,2K//3)`, `[2K//3,K)`) differ by one bin at each edge.
- The official base adapter supports a multi-layer MLP option (`ADAPTER_LAYERS`, `HIDDEN_DIM`); only the
  single-layer linear adapter of the paper's main configuration is implemented.
- Defaults follow the paper and official code: gate initialized to zero (adapted forecast equals the base at
  initialization), SCR Xavier gain 0.01 with zero biases, spectral gate weights zero and bias -1, `r = C`.
  Because the gates start at zero, only the gate parameters and SCR/gate biases receive non-zero gradients
  on the first step.
- No source was copied; see THIRD_PARTY_NOTICES.md.

## Citation

```bibtex
@misc{deng2026correctionspacecrossvariateinteractiontesttime,
  title         = {Correction-space Cross-variate Interaction for Test-time Adaptation in Time Series Forecasting},
  author        = {Yuanyuan Deng and Mykola Pechenizkiy and Songgaojun Deng},
  year          = {2026},
  eprint        = {2609.34638},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2609.34638}
}
```
