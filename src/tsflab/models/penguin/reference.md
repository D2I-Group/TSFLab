# PENGUIN — reference

## Differences in detail

Paper and official code (pinned revision `d9a2b84`, `models/PENGIUN.py`, `layers/SelfAttention_Family.py`, `layers/PENGIUN_EncDec.py`) were consulted for structure only; the official repository has no license file and nothing was copied.

- Period units: the official model passes `periods` straight into the bias as token counts (its scripts use `patch_len=1, stride=1`). The local model takes `periods` in time steps and converts to patch units with `P_S = P / stride` (paper Sec. 4.2.2), requiring every period to be a positive multiple of `stride`.
- Bias options: the official `--alibi` and `--alibicycle` flags are always on here (`alibi=True` adds the bias; periodic distances are used whenever `periods` is non-empty). The official slope construction for head counts that are not a power of two (extra interleaved slopes) is not reproduced; the local slopes are `2^(-8k/n)` per group of `n` heads, which equals the official slopes whenever `n_heads` is a power of two (the default `n_heads=8` with one period).
- `causal` is an added switch (default True; the official always masks). The official `use_rcf` recurrent-cycle option (CycleNet-style cycle removal needing a `cycle_index` input) and the `visual` attention-weight output are not implemented.
- Normalization is the shared `revin` component without affine parameters, equal to the official mean/std normalization (`+1e-5`).
- The feed-forward sub-layer uses Linear layers in place of the official kernel-1 Conv1d (same math).
- `use_rmsnorm` uses the shared RMSNorm (`eps=1e-6`) in place of `LlamaRMSNorm` (computed in the input dtype rather than float32).
- Defaults (`d_model=128`, `d_ff=256`, `e_layers=2`, `patch_len=16`, `stride=8`, `periods=[24]`) differ from the official ETTh1 script (`d_model=16`, `d_ff=128`, `e_layers=3`, `patch_len=1`, `stride=1`, `seq_len=336`, `factor=3`).

## Citation

```bibtex
@article{sun2025penguin,
  title   = {{PENGUIN}: Enhancing Transformer with Periodic-Nested Group Attention for Long-term Time Series Forecasting},
  author  = {Sun, Tian and Chen, Yuqi and Sun, Weiwei},
  journal = {arXiv preprint arXiv:2508.13773},
  year    = {2025}
}
```
