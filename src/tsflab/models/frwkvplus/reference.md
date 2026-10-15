# FRWKVPlus — reference

## Differences in detail

- Implementation: independent rewrite of Sec. 3 (Eqs. 1-22) of arXiv 2605.15690. The official release (`yangqingyuan-byte/FRWKV-plus`, revision `08516fc04e60065d6c1493bbc9cda45c9d0a440c`, AI Scientist Source Code License) was read for `src/adaptive_phasegate_kbs/FRWKV/model/FRWKV_CrossBranchPhaseGateAdaptive.py` (the paper model `frwkv_crossbranchperiodicpositiongate_adaptive`), `FRWKV/model/FRWKV.py`, `FRWKV/layers/Transformer_EncDec.py`, `FRWKV/layers/RevIN.py`, `train.py` and `configs/kbs_ours_recipes.json`; nothing was copied or imported.
- Resolved from the official code: the backbone of FRWKV (`frwkv_linear_attention` branches, `embed_size * freq_bins` tokens per variable, the same head and dropout fractions); PPCE applied to the embedded window before the FFT, padded with its leading steps, bias-free query/key/value/output projections after a LayerNorm, routers initialized `N(0, 0.02^2)`, attention scale `1/sqrt(E)`; Linear-GELU-Linear gate, correction and trust MLPs with zero-initialized correction heads and trust heads with zero weights and constant bias; `alpha` a learnable scalar clamped to `[0, 0.2]` in the forward pass.
- Recipes: `d_model = d_ff = 512`, 8 heads, `embed_size = 16`, 2 layers (3 for Weather), dropout 0.2 (0 for Weather), AdamW with learning rate 1e-4 and weight decay 1e-3, cosine schedule, horizon-weighted L1 with `alpha = 0.5`. `period_len`, `num_routers`, `alpha_init` and `trust_bias_init` default to the Sec. 4.1 values (P = 24, R = 4, 0.10, -2.0); the official recipes tune them per dataset and horizon.
- Shared components, checked against the reference: `frwkv_linear_attention` (extracted with FRWKV) matches the release's `FrequencyRWKVBlock` and frequency branch, which are algebraically identical to `FRWKV@d6dbf17` in the default `joint` / `linear` mode; `revin.RevIN(affine=True)` matches the release's `RevIN`; the encoder layers are `transformer_encdec`'s, matching `EncoderStack` / `EncoderLayer`.

## Verification

- The forward pass was compared with a re-derivation of the official gated forward (PPCE, base gates, signed corrections, trust, clipped budget).
- Checks cover: initialization reduces reduces the gates to `1 + G0` with trust `sigmoid(-2)`, the clipping of `alpha` and the gate bounds `(0.8, 2.2)`, PPCE period averaging with appended leading steps and the cyclic case, and a single-router closed form.
- Also checked: the shared backbone, gradients to the router, budget and trust paths, the horizon-weighted L1 objective, and the strict schema.

## Citation

```bibtex
@article{yang2026frwkvplus,
  title   = {FRWKV+: Periodic-Aware Adaptive Gating for Frequency-Space Linear Time Series Forecasting},
  author  = {Yang, Qingyuan and Chen, Dongyue and Teng, Da and Xiao, Junhua and Pan, Jiaji and Deng, Shizhuo},
  journal = {arXiv preprint arXiv:2605.15690},
  year    = {2026}
}
```
