# DPANet — reference

## Differences in detail

- Sources read: `models/DPANet.py`, `run_longExp.py`, `exp/exp_main.py`, and
  `scripts/*.sh` at revision `f8dbdac23a8134b4d46dc57ed44eb36b312f3e29`.
  Attention uses PyTorch's `nn.MultiheadAttention`, as the official code does.
- Resolved from the official code: RevIN with learnable affine (mean and
  population standard deviation over time, `eps = 1e-5`, statistics detached);
  `num_scales = 3`, `d_model = 128`, `n_heads = 8` (hard-coded), `d_ff = 2048`,
  `dropout = 0.2`; temporal level `s` has length `L // 2^s` (floor pooling for
  odd lengths); the frequency levels keep the full input length.
- Band edges are truncated `exp(u)` values, `u` evenly spaced on `[0, log F]`,
  with the first edge 0 and the last the number of rfft bins; an empty band
  later in the list starts at the previous end.
- Fusion uses post-norm residuals with dropout; the FFN is
  `LayerNorm(fused + Dropout(MLP(fused)))` over the concatenated `2 * d_model`
  states; the head is `Linear(d_model, pred_len)` per channel; training uses
  MSE; calendar marks are not used.
- Validation: `seq_len // 2^(S - 1) >= 1` and `d_model` divisible by `n_heads`.
- The preset follows `scripts/etth1.sh` (ETTh1, `seq_len = 96`, horizon 96,
  batch size 16, learning rate `5e-5`).
- Checked: the band edges and the empty-band
  rule, that frequency levels partition the signal and stay in their bands, the
  pairwise-mean temporal levels, the fusion block against a manual evaluation of
  Eqs. 3-4 and the FFN (one and several tokens), the one-token attention
  reducing to the value projection, layer sizes, the full forward pipeline
  against a manual composition with non-trivial affine RevIN, shift
  equivariance, odd lengths and a single scale, gradient flow, invalid
  configurations, and the parameter schema. Reported benchmark numbers are not
  reproduction claims of this implementation.

## Citation

Li, Q., Zhang, X., Wang, S., Jia, W. "DPANet: Dual Pyramid Attention Network for Multivariate Time Series Forecasting." arXiv:2509.14868 (2025).
