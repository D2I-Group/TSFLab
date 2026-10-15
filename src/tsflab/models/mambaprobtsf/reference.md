# MambaProbTSF — reference

## Paper

Pessoa, Campitelli, Shepherd, Ozkan, Pressé. Machine Learning: Science and Technology 6, 035012 (2025),
doi:10.1088/2632-2153/adec3b; arXiv 2503.10873 (2025-03).

## Differences in detail

**Resolved from the official code** (`model/S_Mamba.py`, `layers/Mamba_EncDec.py`,
`experiments/exp_probabilistic_forecasting.py`, `run_prob.py`, `ECL_script.sh` at revision `a69d7c3c`):
the mean network is S-Mamba with `Mamba(d_model, d_state, d_conv = 2, expand = 1)` forward and flipped scans
whose sum is added without dropout, post-norm LayerNorms, a GELU pointwise feed-forward network with dropout,
a final LayerNorm, the inverted embedding with appended calendar tokens, and non-stationary instance
normalization. The sigma MLP (`sigma_network Linear`, `sigma_method General`) reads the raw, unnormalized
window per variate with hidden width 512 and GELU, and adds `1e-8` after the softplus; with the S-Mamba sigma
network the softplus is applied to the denormalized S-Mamba output. The loss is
`((y - mu) / sigma)^2 / 2 + log sigma` averaged over batch, horizon, and channels. Defaults `d_model = 512`,
`d_ff = 512`, `d_state = 16`, `e_layers = 3` follow the reported scripts.

**Loss constant.** The catalog `nll_gaussian` loss adds `0.5 log(2 pi)` and floors the scale at `1e-6`, which
does not change gradients for `sigma > 1e-6`.

**Calendar tokens** are built from raw marks with `marks.adapt_tslib_marks`, which supports hourly `timeF`
only; the paper's synthetic datasets carry no meaningful calendar.

**Verified properties:** the `(loc, scale)` output contract and positivity, that
the location is the S-Mamba mean network and the scale the sigma MLP on the raw window (GELU layers, softplus
plus `1e-8`), the bidirectional scan and post-norm order of the S-Mamba layer built from the cataloged
`MambaBlock`, the S-Mamba sigma variant, that the likelihood of Eq. (8) equals the catalog `nll_gaussian` loss
up to its constant, that gradients reach both networks, calendar-token handling, RevIN shift equivariance of
the mean, and the parameter schema. Reported benchmark numbers are not reproduction claims.

## Citation

```bibtex
@article{pessoa2025mamba,
  title   = {Mamba time series forecasting with uncertainty quantification},
  author  = {Pessoa, Pedro and Campitelli, Paul and Shepherd, Douglas P. and Ozkan, S. Banu and Press{\'e}, Steve},
  journal = {Machine Learning: Science and Technology},
  volume  = {6},
  pages   = {035012},
  year    = {2025},
  doi     = {10.1088/2632-2153/adec3b}
}
```
