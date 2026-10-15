# GCformer — reference

## Differences in detail

- Implementation: independent rewrite from Section 3 (Eqs. 1-3 and 8-15, Fig. 3) of the paper after reading the pinned official code (`zyj-111/GCformer`, revision `67776069a34700929117618f8541e3880f396b7b`; no license file, recorded as `NOASSERTION`): `models/GCformer.py`, `layers/global_conv.py` (`GConv`), `layers/PatchTST_backbone.py`, `layers/PatchTST_layers.py`, `layers/SelfAttention_Family.py`, `layers/RevIN.py`, `exp/exp_main.py`, `run_longExp.py`, and `scripts/GCformer/*.sh`. Nothing was copied or imported.
- Global branch from the code: `GConv` in its default `cat_randn` mode (randomly initialised 32-tap sub-kernels per head and variate, linear upsampling by `2 ** max(0, i - 1)`, weights `2 ** (n - i - 1)` with decay multiplier 2, kernel normalised by its initial norm, bidirectional with `n_heads` causal and `n_heads` anti-causal kernels, `D` skip, GELU, a linear output map from `n_heads * enc_in` to `enc_in`) followed by `Linear(seq_len, pred_len)`.
- Local branch from the code: PatchTST without its own RevIN (`local_revin = 0`), end padding, `zeros` learnable position table, BatchNorm, residual attention, per-channel heads (`individual = 1`).
- Decoder from the code: computes both query directions, adds the key/value input as residual, shares one BatchNorm per view, applies GELU + dropout (`fc_dropout`) around the projections, mixed by `atten_bias` and `TC_bias`. `global_bias` and `local_bias` are learnable scalars initialised to the configured value plus `U(0, 0.1)`; the outer RevIN is affine; training uses MSE on the final output only.
- Preset: `scripts/GCformer/illness.sh` for horizon 24 (`seq_len = context_len = 104`, `patch_len = 24`, `stride = 2`, `d_model = 16`, `n_heads = 4`, `d_ff = 128`, `e_layers = 3`, `dropout = fc_dropout = 0.2`, shortcuts 0.3, default `h_channel = 32`, `h_token = 512`, `TC_bias = 1`).
- Decoder attention: the official decoder uses Informer ProbSparse attention with its causal mask and cumulative-sum context and passes the `[B, H, L, D]` context to `out.view(B, L, -1)` in `AttentionLayer` (`layers/SelfAttention_Family.py`), which interleaves heads and positions; `decoder_attention = "prob"` uses the cataloged ProbSparse core with the head layout corrected.
- `GConv`: the norm is fixed at construction instead of at the first forward call (identical values because no update happens in between); its unused batch-size-shaped parameters are not created; the official sub-kernel count reaches zero for windows shorter than 32 steps.
- Not implemented: the unused modules instantiated by the official model (TCN, an Autoformer, `linear_local_token`); the auxiliary `local_x`, `global_x`, and bias outputs of the official forward are not returned.

## Verification

Checked properties: the sub-kernel count and assembled length; the decaying scale weights; the FFT convolution against a direct two-sided sum with skip, GELU, and output map; the unit initial kernel norm; the end-padded patches; channel independence and residual attention scores of the local branch; both query directions of the cross fusion against a manual computation; the ProbSparse option; the full forward composition with the shortcuts; the local branch reads only the recent context; last-value normalization equivariance; learnable shortcut weights. Reported benchmark numbers are not reproduction claims.

## Citation

```bibtex
@misc{zhao2023gcformer,
  title         = {GCformer: An Efficient Framework for Accurate and Scalable Long-Term Multivariate Time Series Forecasting},
  author        = {Zhao, YanJun and Ma, Ziqing and Zhou, Tian and Sun, Liang and Ye, Mengni and Qian, Yi},
  year          = {2023},
  eprint        = {2306.08325},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG}
}
```
