# TimeMachine — reference

## Paper

TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting (Ahamed, Cheng; ECAI 2024;
arXiv:2403.09898).

## Differences in detail

- Official files read at the pinned revision: `TimeMachine_supervised/models/TimeMachine.py`, `RevIN/RevIN.py` and the dataset scripts.
- Interface: inputs `[B, seq_len, enc_in]`; marks and decoder inputs are ignored; output `[B, pred_len, enc_in]`.
- Mamba kernels: official code uses `mamba_ssm.Mamba` with fused CUDA kernels; here the four mixers are the `mamba` component's sequential PyTorch recurrence with the same selective-scan math, a causal depthwise convolution of width `d_conv` with SiLU, `dt_rank = ceil(width / 16)`, and no norm or residual inside the mixer.
- Official Mamba initialises the step-size projection with its reference scheme (uniform weight, log-uniform step bias); the mixers here pass `reference_dt_init=True` to match it.
- Normalization: `revin=True` is RevIN with affine parameters; `revin=False` keeps the instance mean and standard deviation (biased, eps 1e-5) but drops the learnable affine, as the official fallback does.
- Preset values: official scripts set `n1`/`n2` and `fc_drop` per dataset and horizon (ETTh1 uses `n1` in {128, 512} and `fc_drop=0.7`). `d_state=256`, `d_conv=2`, `expand=1`, `ch_ind=True` and `residual=True` match the scripts; the Traffic script disables RevIN (`rin=0`).
- Dataflow: the axis swaps, the two residual links (`n2` embedding before `proj1`, `n1` embedding after it) and the concatenation with the outer pair follow the official forward pass exactly.
- Training: loss, optimiser, schedule and early stopping are runner configuration.

## Citation

```bibtex
@inproceedings{ahamed2024timemachine,
  title     = {TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting},
  author    = {Md Atik Ahamed and Qiang Cheng},
  booktitle = {27th European Conference on Artificial Intelligence (ECAI)},
  year      = {2024},
  url       = {https://arxiv.org/abs/2403.09898}
}
```
