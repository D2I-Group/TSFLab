# SAMBA — reference

## Differences in detail

- Implementation: local rewrite from Sections 5.1 and 5.2. The official code (`YukinoAsuna/SAMBA`, MIT) was inspected at revision `490257dac0806e16c00372226870850d39e8be65` (`model/SAMBA.py`, `layers/Block.py`, `layers/mixer_seq_simple.py`, `layers/samba_simple.py`, `scripts/SAMBA/`) to resolve omissions; no source was copied.
- Resolved from the official code: replication padding of one stride before unfolding, a bias-carrying patch projection `Linear(patch_len, d_model)`, a learnable `[patches, 1]` position term added only on the time branch.
- Blocks: `Add -> LayerNorm -> mixer -> Add -> LayerNorm -> gated MLP` returning `(hidden, residual)`; the shared stack norm is applied to the residual after every block and to `hidden + residual` at the end.
- Fusion: `Linear(2d, d) -> GELU -> Dropout -> Linear(d, d)`; a single flatten head shared across variates; the step-size initialization of the reference Mamba (`dt_proj`).
- The gated MLP follows `mamba_ssm==2.2.2`: SiLU gate and hidden width rounded up to a multiple of 128.
- Kernels: the selective scan, depthwise convolution and gating run through the shared pure-PyTorch `mamba` component (a sequential Python-loop scan) instead of the CUDA `mamba_ssm` kernels and `causal_conv1d`, so it is portable but much slower; `use_act=True` applies SiLU after the convolution as the official kernel path does.
- Initialization: other weights use PyTorch defaults, as the official encoders are built without the `MixerModel` initializer.
- Preset: `d_model=128`, `d_ff=128`, `e_layers=1`, `d_layers=1`, whereas the official script for Electricity uses `d_model=512`, `d_ff=512`, `e_layers=1`, `d_layers=2`, learning rate 5e-4, batch size 16 (the `run.py` defaults are 512, 128, 2 and 1); per-dataset settings are not preset.
- `e_layers=0` or `d_layers=0` keeps a single branch without the fusion MLP, as in the official code.
- Structure is checked locally; no training was run and the CUDA kernels were not compared numerically. Reported benchmark numbers are not reproduction claims of this implementation.

## Paper

SDE: A Simplified and Disentangled Dependency Encoding Framework for State Space Models in Time Series Forecasting, KDD 2025 (arXiv 2408.12068, 2024-08). The paper proposes the SDE framework and its instantiation SAMBA (SDE-Mamba); this entry implements SAMBA.

## Citation

```bibtex
@inproceedings{weng2025sde,
  title         = {{SDE}: A Simplified and Disentangled Dependency Encoding Framework for State Space Models in Time Series Forecasting},
  author        = {Weng, Zixuan and Han, Jindong and Jiang, Wenzhao and Liu, Hao},
  booktitle     = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year          = {2025},
  eprint        = {2408.12068},
  archivePrefix = {arXiv},
  url           = {https://arxiv.org/abs/2408.12068}
}
```
