# DMSC — reference

## Paper

- **Title**: DMSC: Dynamic Multi-Scale Coordination Framework for Time Series Forecasting
- **Venue**: arXiv preprint (2025)
- **arXiv**: https://arxiv.org/abs/2508.02753

## Implementation details

- Independent rewrite of Section 3 (Eqs. 1-23) and Table 2 of arXiv 2508.02753. The official repository
  (`1327679995/DMSC` at `57d03f11`, no license, `NOASSERTION`) was read as reference only (`models/DMSC.py`,
  `layers/DMSC_Layer.py`, `layers/StandardNorm.py`, `experiments/exp_forecasting_DMSC.py`, `run.py`); nothing
  was copied or imported.
- EMPD dynamic scale (Eqs. 6-10): `ScaleFactorNet` maps the variate-averaged batch to `alpha`,
  `P_base = P_min + alpha (P_max - P_min)`, layer `l` uses `max(P_min, P_base / tau^l)`; `unfold_patches` and
  `PatchEmbedding` replicate-pad, unfold with stride `P // 2`, pool every patch to 8 samples and project it.
- `TriadInteractionBlock` (Eqs. 11-16): per-variate depth-wise separable intra-patch conv, dilated inter-patch
  conv with pooling, a cross-variable sigmoid gate from the variate-averaged feature, softmax fusion of the
  three, and a LayerNorm residual from the patch-averaged embedding.
- `Model.cascade` (Eqs. 1-5): layer `l > 1` sees the input plus a sigmoid-gated projection of the previous
  scale feature back to the input length.
- `AdaptiveScaleRoutingMoE` (Eqs. 17-21): one softmax gate routes each scale feature to always-on global
  experts (depth 3) and the top-k local experts (depth 2); scale forecasts are fused with softmax weights from
  forecast descriptors and a learnable history vector `w_hist`; a seq-to-horizon MLP residual is added before
  RevIN denormalization. `balance_loss` is Eq. 22; `training_objective` adds it to the forecast loss (Eq. 23).
- Resolved from the official code: scale net (3-tap conv to 4 channels, GELU, global pooling, 4-16-32-1 GELU
  MLP, sigmoid, dropout, batch mean); TIB layer order (depth-wise conv, BatchNorm, GELU, pointwise conv,
  BatchNorm, dropout, GELU; dilation 2 for the inter-patch branch); fusion gate
  `Linear(3D, D) -> GELU -> Linear(D, 3) -> softmax`; cascade gate `Linear -> GELU -> Linear(D, 1) -> sigmoid`
  times `Linear(D, seq_len)`; global experts `Linear-GELU-Linear-GELU-Dropout-Linear`, local experts
  `Linear-GELU-Dropout-Linear`, router hidden `2 D`; scale weights `Linear -> Tanh -> Softplus -> softmax`;
  output `Linear-GELU-Linear-Dropout`; affine RevIN.
- Shared components: `revin.RevIN(affine=True)` matches `layers/StandardNorm.Normalize` except for the `1e-10`
  added to the affine weight on denormalization; `topk_expert_router.GatingMLP(noisy=False)` matches the router.
  `topk_dense_mix` is not reused because DMSC does not renormalize the kept local weights.

## Differences in detail

- Paper/code conflicts and their resolutions are recorded as `issues` in `card.toml`.
- The `--use_res` per-scale expert residual (off by default) is not implemented.
- Training uses the catalog trainer with the configured loss plus the balance loss (the paper uses MSE, Adam,
  learning rate 1e-3, batch 32). Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@article{yang2025dmsc,
  title   = {DMSC: Dynamic Multi-Scale Coordination Framework for Time Series Forecasting},
  author  = {Yang, Haonan and Tang, Jianchao and Li, Zhuo and Lan, Long},
  journal = {arXiv preprint arXiv:2508.02753},
  year    = {2025}
}
```
