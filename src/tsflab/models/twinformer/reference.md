# TwinFormer — reference

## Paper

TwinFormer: A Dual-Level Transformer for Long-Sequence Time-Series Forecasting (arXiv 2512.12301, 2025). Sources used: Sec. 3.1-3.2, Eqs. 1-18, Sec. 4.2-4.3, Fig. 1.

```bibtex
@article{kumavat2025twinformer,
  title   = {TwinFormer: A Dual-Level Transformer for Long-Sequence Time-Series Forecasting},
  author  = {Kumavat, Mahima and Maheshwari, Aditya},
  journal = {arXiv preprint arXiv:2512.12301},
  year    = {2025}
}
```

## Implementation mapping

Independent rewrite after reading the pinned official code (`Mahimakumavat1205/TwinFormer` at `c034df43`, no license file): `Scripts/main.py` (`TopKSparseAttention`, `SparseBlock`, `SinusoidalPE`, `I3InformerV2` and its per-dataset copies) and `README.md`. Nothing was copied or imported. The repository ships one concatenated notebook export of per-dataset scripts for `I3InformerV2` (no `TwinFormer` class or entry point), so the paper's equations define the model and the script only resolves omitted details.

Resolved from the official code:

- Biased Q/K/V/output projections with head width `d / h`; top-k via `torch.topk` scattered into a `-inf` mask; no dropout on attention weights.
- Dropout on both residual branches and inside the feed-forward; a sinusoidal position table (with dropout) added before patching, since the paper defines no positions and mean pooling would otherwise ignore step order within a patch.
- `torch.nn.GRU` with zero initial state (reset gate after the hidden projection, update gate in torch's convention, an equivalent reparameterization of Eq. 17).
- Unspecified sizes from the code: 4 heads, `4d` GELU feed-forward, dropout 0.15, `P = 12`, `d = 128` (Fig. 1 shows 32-dimensional tokens).
- Output `Linear(d -> H * C)` for all channels, as the ETTh1 and Weather sections map to `pred_len` times the target channels (Eq. 18 writes one series). Sec. 3.2.5 writes the final state as `R^{Np x d}`; Eq. 18 uses one `d`-vector, which is used here.
- The preset follows the paper's input length 48 and horizon 96 in its contract fixture.

## Differences in detail

- Pooling: Eq. (10) mean-pools each patch; the code keeps the last token (`p[:, -1, :]`). Mean pooling is used.
- Depth: the paper has one Local and one Global block; the code two each. Default one; `local_layers` and `global_layers` expose the code's depth.
- Normalization: the code puts LayerNorm before attention as well as the feed-forward; the paper's residual order (Eqs. 8, 12) is used.
- Head: the code uses a two-layer GRU averaged over patches, LayerNorm, a three-layer GELU MLP head, and a linear skip; the paper's single GRU final state and linear layer are used.
- Sparsity: Eq. (2) keeps `k = 5`; the code sets `TOPK = 2`.
- Scaling: the paper uses min-max, the code `StandardScaler` fitted on the whole series including validation and test (leakage); TSFLab uses the runner's train-split scaling.
- Inputs: the code augments with cyclical hour and weekday features and lag-24/lag-48 target copies; only raw channels are used here.
- Trimming: the code drops the most recent `L mod P` steps; the oldest are dropped here.
- Training: the code's `CombinedLoss` (MAE+MSE), warm-up cosine schedule, and gradient clipping are not reproduced; the paper trains with MSE.
