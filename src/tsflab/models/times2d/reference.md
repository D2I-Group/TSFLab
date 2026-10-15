# Times2D — reference

## Paper

Times2D: Multi-Period Decomposition and Derivative Mapping for General Time Series Forecasting
(Nematirad, Pahwa, Natarajan; AAAI 2025; arXiv:2504.00118).

## Implementation mapping

- `period_grid` (Eq. 3): right zero-padding to whole cycles of one period and to a multiple of that period's `patch_len` in columns.
- `PeriodBranch` (Eqs. 4-10): `Conv2d(1, rows * patch_len)` with kernel and stride `rows x patch_len`, linear embedding, learnable position table (`positional_encoding("zeros", learnable)`, uniform +-0.02), `e_layers` post-norm `TokenEncoderLayer`s (residual-attention scores carried across layers, BatchNorm, GELU feed-forward), flatten feature-major, linear map to `seq_len`.
- `fsdh` (Eqs. 11-16): two `InceptionBlock2d` (cataloged `inception_block`, Kaiming fan-out init, `num_kernels = 6`, width `d_ff`) with GELU between, derivative axis summed with two learned weights, linear map over time shared across channels.
- The residual-attention Transformer layer is model-local: the cataloged `patchtst`/`tst_transformer` encoders use `torch.nn.TransformerEncoderLayer` without residual attention scores.

## Differences in detail

- Official code read at `Tims2D/Times2D` revision `ab819a00c8ebe446b7954897470f86508032f7a7`: `models/Times2D.py`, `layers/encoders.py`, `layers/attention.py`, `layers/All_layers.py`, `layers/Conv_Blocks.py`, `layers/RevIN.py`, `run.py`, `scripts/times2D/longTerm/Times2D_ETTh1.sh`. Its custom `LICENSE` places the code under AGPLv3 (no "or later" clause) with third-party MIT/Apache parts.
- Resolved from the official code: periods are a fixed list with one patch width each (ETTh1: `seq_len = 720`, periods `720, 360, 110, 96, 48`, widths `48, 32, 16, 6, 3`); encoder `e_layers = 3`, `n_heads = 16`, `d_model = 64`, `d_ff = 64`, dropout 0.5, attention dropout 0.05, `head_dropout = 0`; heatmap convolutions are TimesNet-style inception blocks whose input maps are the channels; RevIN is non-affine with mean subtraction.
- The paper selects the top-k FFT periods per batch (Eqs. 1-3); the official `FFT_for_Period` call is commented out and a fixed list is hard-coded because the patch convolutions and linear maps are sized per period.
- The official derivative weights are `nn.Parameter(torch.randn(batch, 2))` (one pair per batch row, so only full batches work); the M4 branch draws fresh `torch.randn(B, 2)` at every forward pass, so those weights are never trained.
- Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@inproceedings{nematirad2025times2d,
  title     = {Times2D: Multi-Period Decomposition and Derivative Mapping for General Time Series Forecasting},
  author    = {Nematirad, Reza and Pahwa, Anil and Natarajan, Balasubramaniam},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025}
}
```
