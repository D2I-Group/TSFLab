# VCformer — reference

## Paper

VCformer: Variable Correlation Transformer with Inherent Lagged Correlation for Multivariate Time Series Forecasting (IJCAI 2024 Main Track, pages 5335-5343; arXiv 2405.11470).

```bibtex
@inproceedings{yang2024vcformer,
  title     = {VCformer: Variable Correlation Transformer with Inherent Lagged Correlation for Multivariate Time Series Forecasting},
  author    = {Yang, Yingnan and Zhu, Qingling and Chen, Jianyong},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence (IJCAI-24)},
  pages     = {5335--5343},
  year      = {2024},
  doi       = {10.24963/ijcai.2024/590}
}
```

## Implementation mapping

Independent rewrite from Section 3 (Eqs. 3-11) and Algorithm 1 after reading the pinned official code (`CSyyn/VCformer` at `67e8dc8c`; no license file, recorded as `NOASSERTION`): `models/VCformer.py`, `layers/VCformer_Enc.py`, `layers/KTD.py`, `layers/SelfAttention_Family.py` (`VarCorAttention`, `VarCorAttentionLayer`), `layers/Embed.py`, `run.py`, and `scripts/VCformer_scripts/ETT_scripts/ETTh1.sh`. Nothing was copied or imported.

Resolved from the official code:

- Lags run over the `d_model` axis of the projected variate tokens; the correlation is `irfft(F(q) * conj(F(k)))` and the softmax is scaled by `1 / sqrt(d_model)`.
- The VCA is a single head over the full width (`n_heads` does not split it) with no attention dropout.
- The encoder layer has no feed-forward network (the KTD replaces it); dropout is applied to the VCA and KTD outputs before each post-norm residual.
- KTD encoder and decoder are per-token MLPs (`Linear`, tanh, dropout 0.05, `Linear`) over snapshots of length `snap_size`; a non-finite Koopman operator is replaced by the identity.
- Calendar features are appended as extra tokens and dropped before the output, built from raw marks with `marks.adapt_tslib_marks`.
- Instance normalization is non-stationary-style (mean and biased standard deviation plus `1e-5`).
- Defaults `d_model = 512`, `e_layers = 3`, `dropout = 0.1`, `snap_size = 16`, `proj_dim = 128`, `hidden_dim = 256` follow the ETTh1 script and `run.py`.

## Differences in detail

- Eq. (5): the paper's lag weights `lambda` are learnable; the official code takes `corr.mean(dim=-1)` (fixed `lambda = 1 / d_model`). TSFLab learns them from that initial value.
- Eq. (8): the operator is the Moore-Penrose solution (`torch.linalg.pinv`), the minimum-norm least-squares solution the official `torch.linalg.lstsq` returns with its default CPU driver.
- Eq. (9): the last snapshot embedding is advanced by `K^t` for `t = 1..d_model / snap_size`; the official `KPLayerApprox` advances every snapshot embedding by `K^(d_model / snap_size)`, predicting the same positions from different starting snapshots.
- Padding: the official `KTDlayer` left-pads tokens whose width is not a multiple of `snap_size`, breaking the residual; here `snap_size` must divide `d_model`.
- Snapshot size: the appendix mentions `S = 32`; the scripts use the `run.py` default 16, which is the default here.
