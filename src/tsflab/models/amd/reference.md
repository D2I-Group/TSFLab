# AMD — reference

## Citation

```bibtex
@inproceedings{hu2025adaptive,
  title     = {Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting},
  author    = {Hu, Yifan and Liu, Peiyuan and Zhu, Peng and Cheng, Dawei and Dai, Tao},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2025}
}
```

## Differences in detail

Paper and official code (pinned revision `000d377`, `models/tsAMD.py`, `models/common.py`, `models/tsmoe.py`) were checked; the local module is an independent rewrite. MDM, DDI, the noisy gate (log1p below the k-th logit, expm1 above, scaled by 10, then softmax) and the importance loss follow the official modules.

- The official `AMS` loops over channels in Python with one shared gate and one shared expert set; the local module computes all channels in one batched pass (same math, same shared parameters). The gate input is the MDM time embedding and the experts read the DDI output, as in the official model.
- The importance loss reproduces the official `cv_squared`: gates are expanded over the horizon before the batch sum, so the variance uses the `experts * pred_len` element count, summed over channels. It is exposed as `model.aux_loss` (scaled by `moe_loss_coef`, default 1.0 like the official `loss_coef`) in training mode only and added to the criterion by the shared trainer; it is `None` in evaluation. The official `forward` returns `(forecast, moe_loss)` and its loop adds the loss only in training.
- Official hard-codes `ff_dim=2048`, `num_experts=8`, `top_k=2` and calls the BatchNorm switch `layernorm`; TSFLab exposes them as `ff_dim`, `num_experts`, `top_k` and `batchnorm` (BatchNorm1d over flattened `channels * length` features, so `enc_in` and `seq_len` are fixed at construction). The official `target_slice` output selection is dropped.
- Shape validation: `seq_len` must be divisible by `patch` and by `mix_layer_scale**mix_layer_num`, and `top_k <= num_experts`. The official scripts use `seq_len=512` while the contract fixture uses 96; the preset values (`n_block=1`, `alpha=0.0`, `mix_layer_num=3`, `patch=16`, `dropout=0.1`) follow the official scripts.
- RevIN is the shared `revin` component with affine parameters, matching the official RevIN.
