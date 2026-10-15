# SCFormer — reference

## Differences in detail

- The official repository (`ShiweiGuo1995/SCFormer`, revision `a6f5aecfcf375833d3a4c6760f01d467e0445c38`; the model is `SiTransformer` there, per `readme.txt`) has no license file, so it is used as reference only: `model/SiTransformer.py`, `layers/SelfAttention_Family.py`, `layers/Transformer_EncDec.py`, `utils/hippo.py`, `data_provider/data_loader.py`, `run.py` and `scripts/multivariate_forecasting/` were read to resolve omissions; nothing was copied or imported.
- Resolved from the official code: LegS with bilinear discretization (`A_t = (I - A/2t)^-1 (I + A/2t)`, `B_t = (I - A/2t)^-1 B/t`) applied to loader-scaled (not instance-normalized) values, the state taken at the first look-back step.
- The window is instance-normalized without affine parameters (mean detached, standard deviation with `1e-5`).
- `Linear(lookback, d_model)` window embedding, `Linear(hippo_order + d_model, d_model)` mixing, and a per-channel embedding initialized uniformly in `[-0.02, 0.02]`.
- Full (unmasked) scaled dot-product attention over channel tokens with attention dropout; post-attention residual, LayerNorm before the feed-forward block and after its residual; final encoder LayerNorm and `Linear(d_model, pred_len)` head.
- Official scripts: `d_model` 256-1024, 2-4 layers, 16 heads, GELU, dropout 0.1, triangular mode by default and convolutional mode for ETTm1, ETTm2 and Traffic (kernel 32).
- Shared component check: `self_attention_family.FullAttention(mask_flag=False)` matches the official `FullAttention` (scale `1/sqrt(E)`, softmax, dropout, einsum layout).
- The history must be part of the input window (`seq_len >= lookback`). Official code recomputes LegS over a sliding 2048-step window (256 for PEMS) and uses a shorter prefix near the series start; catalog loaders do not produce windows without enough preceding data.
- Training uses the catalog trainer and loss (the paper uses MSE; scripts use learning rates 5e-4 to 1e-3). Reported benchmark numbers are not reproduction claims of this implementation.
- Not reproduced: the official evaluation writes `ts_time.npy` on every forward, and its HiPPO cache is keyed only by data file name; here coefficients are computed inside the model.

## Citation

```bibtex
@article{guo2025scformer,
  title   = {SCFormer: Structured Channel-wise Transformer with Cumulative Historical State for Multivariate Time Series Forecasting},
  author  = {Guo, Shiwei and Chen, Ziang and Ma, Yupeng and Han, Yunfei and Wang, Yi},
  journal = {arXiv preprint arXiv:2505.02655},
  year    = {2025}
}
```
