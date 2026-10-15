# CAPS — reference

## Paper

CAPS: Unifying Attention, Recurrence, and Alignment in Transformer-based Time Series Forecasting (arXiv 2602.02729v1, 2026-02; Sec. 2.5, 3.1-3.4, Eqs. 7-18, Fig. 2, Sec. A.2-A.3).

```bibtex
@article{pati2026caps,
  title   = {CAPS: Unifying Attention, Recurrence, and Alignment in Transformer-based Time Series Forecasting},
  author  = {Pati, Viresh and Kim, Yubin and Pham, Vinh and Twitty, Jevon and Yang, Shihao and Lu, Jiecheng},
  journal = {arXiv preprint arXiv:2602.02729},
  year    = {2026}
}
```

## Implementation mapping

- `CAPSAttention.clock_weights` is Eq. (7). `CAPSAttention.path_queries_keys` builds Eq. (16): Path 1 `q / sum_{j<=t} e^{p~_j}`, `k e^{p~_i}` (Eqs. 9-11); Path 2 `q Gamma_t`, `k / Gamma_i` with `log Gamma_t = sum_j -softplus(g_j) Delta_j` (Eqs. 12-14); Path 3 `q / sum_{j<=t} Delta_j`, `k Delta_i` (Eq. 15). `rope_rotate` is Eq. (8).
- `causal_linear_attention` is Eq. (17): `o_t = sum_{i<=t} (q_cat,t . k_cat,i) v_i`.
- `Model.extend` is Eq. (18); `Model.tokens` forms the dual tokens; `CAPSBlock` is pre-RMSNorm attention plus a GELU FFN; `Model.forward` decodes `y_{c,t} = h_{c,t} V_c^T` from the value slice.
- Official files read (nothing copied): `models/caps.py` (`SimpleRoPE`, `RMSNorm`, `MLP`, `CAPSAttention`, `Transformer`, `random_ratio_channel_dropout_vectorized`, `Model`), `caps.sh`, `run_longExp.py`, `exp/exp_main.py` (`vireshpati/CAPS-Attention`, revision `e6263fd4`).

## Differences in detail

- Resolved from the official code: bias-free projections; per-feature Path 1 and Path 2 rates with a per-head Clock broadcast over the head width; no `1/sqrt(d)` scaling; causal (lower-triangular) linear attention over the extended `L + H` sequence; pre-RMSNorm blocks with a `4x` GELU FFN and dropout on both residual branches; decoding from horizon tokens only; normal(0, 0.02) initialization with the `c_proj` scaling `1 / sqrt(2N)` (`out_fill` keeps PyTorch's default); learned position table over `L + H` steps and per-channel embedding (both zero init); random-ratio channel dropout on the channel-token input only, survivors divided by the realised keep fraction.
- RoPE: each pair is rotated by one angle `t w_l` with learned frequencies initialized at base 10000 (`learn_rope_freq = false` keeps them fixed); the official layout turns pair members by different angles.
- Numerics: the -50 floor on the cumulative log-gate is kept as a guard (Eq. 14 holds exactly while the prefix log-decay stays above -50); the global max shift of Path 1 is kept with the denominator clamped to the smallest positive float.
- Inputs: the official scripts append four calendar features (last-value shifted, window-std scaled) to the channel token; here marks are ignored.
- Widths: Sec. A.2 states 64 + 64 (8 for ETTh), the scripts derive 16 + 16 for ETT; Table 4's 527K parameters on ETTm1 do not match the scripted ~59K. Defaults follow the script.
- Training uses the configured loss (the paper trains with MSE) and the catalog optimizer and schedule (weight decay is run configuration); checkpoints are selected on validation only (the official run also logs per-metric minimum test scores).
- Checked: RoPE norm preservation and offset-only scores, path products against a direct evaluation of Eqs. (11), (14), (15), causality, the forward composition, last-value shift equivariance, and channel-dropout rescaling. Reported benchmark numbers are not reproduction claims of this implementation.
