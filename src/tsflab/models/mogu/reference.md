# MoGU — reference

## Paper

- **Title**: MoGU: Mixture-of-Gaussians with Uncertainty-based Gating for Time Series Forecasting
- **Venue**: arXiv preprint 2510.07459 (2025; v2 February 2026)

## Differences in detail

**Implementation.** Independent rewrite of Sections 3.2-3.4 (Eqs. 6-15) and
Listing 1. The official repository (`yolish/moe_unc_tsf`, MIT) was read at the
pinned revision to resolve omissions: `models/MoE.py`, `models/iTransformer.py`,
`exp/exp_long_term_forecasting.py`, `run.py`, and `scripts/run_tsf_ett.sh`;
nothing was copied or imported.

**Resolved from the official code.**

- The gate is computed elementwise per expert, horizon step, and variable with `1 / (sigma^2 + 1e-8)`.
- The loss multiplies each expert's elementwise Gaussian NLL (`0.5 * (log var + residual^2 / var)`, variance clamped at `1e-6`, no constant) by its weight, sums over experts, and averages over all elements.
- The uncertainty head is `Linear(d_model, d_model) -> ReLU -> Linear(d_model, pred_len)` on the encoder output, then `softplus` (threshold 20); `head_type = "linear"` drops the hidden layer.
- The expert uses instance normalization without affine parameters (mean detached, `1e-5` in the standard deviation), the inverted embedding with dropout, a post-norm encoder with a final LayerNorm, and `Linear(d_model, pred_len)`. Means are de-normalized with the window statistics.
- The aleatoric term `sum_i w_i sigma_i^2` equals `k / sum_j sigma_j^-2`, the harmonic mean of the expert variances.

**Shared components.** `embed.DataEmbedding_inverted` (linear `seq_len -> d_model`
per variate plus dropout), `self_attention_family.FullAttention(mask_flag=False)`
with `AttentionLayer`, and `transformer_encdec.Encoder`/`EncoderLayer` (conv-1x1
feed-forward, post-norm) match the official `layers/Embed.py`,
`layers/SelfAttention_Family.py`, and `layers/Transformer_EncDec.py`.

**Other differences.** Calendar marks are not used (the official TSLib
iTransformer appends `timeF` mark tokens to the variate tokens). The
predictive distribution is moment-matched to one Gaussian. The variance clamp
blocks the gradient below `nll_eps`. Only the iTransformer expert is provided.
The paper's training settings (Adam, 10 epochs, patience 3, learning rate 1e-4,
1e-3 for Weather and Electricity) are run settings, not model parameters.
Reported benchmark numbers are not reproduction claims of this implementation.

**Checked properties.** The Eq. (10) precision gate; the Eq. (7)/(12) moments
against a hand computation (including the harmonic-mean aleatoric term); the
Eq. (6) clamped NLL; the Eq. (11) loss against an explicit per-expert sum; the
uncertainty head architecture and positivity; independent experts built from
the shared components; the `[B, H, C, 2]` distribution contract; gradients of
the training objective reaching every expert's mean and variance paths;
`features = MS` alignment; `denorm_variance`; and the strict parameter schema.

## Citation

```bibtex
@article{aviv2025mogu,
  title   = {MoGU: Mixture-of-Gaussians with Uncertainty-based Gating for Time Series Forecasting},
  author  = {Aviv, Gilad and Goldberger, Jacob and Shavit, Yoli},
  journal = {arXiv preprint arXiv:2510.07459},
  year    = {2025}
}
```
