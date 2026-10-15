# SEMixer — reference

## Paper

ACM Web Conference 2026 (WWW 2026), arXiv:2602.16220. SEMixer is an end-to-end lightweight multiscale model. RAM learns diverse patch interactions by random sampling during training and aggregates them via a dropout-ensemble approximation at inference. MPMC stacks RAM and the MLP-Mixer backbone as the scale level increases and restricts mixing to pairwise concatenation of adjacent scales. Validated on 10 public datasets and a real-world wireless-network dataset from the 2025 CCF AIOps Challenge (third place).

## Differences in detail

- The official code hardcodes four scales via dict keys `{1, 2, 4, 8}` and per-scale attributes (`W_pos_1`..`W_pos_4`, `T_Mixing_Scale_1`..`_4`). The local MPMC uses `nn.ModuleList` chains over an arbitrary strictly increasing schedule starting at 1 (default `"1x2x4x8"`, the paper's Sec. 4.1.3 default); mathematically identical for four scales.
- The official `Flatten_Head` supports `individual` per-channel and `var_decomp` variable-group heads selected by CLI flags never exercised in the paper's reported configuration; the local head is always shared.
- The official code exposes optional non-RAM backbones (`ProbAttention`, `LogSparseAttention`, `PerformerLayer`, `ReformerLayer`, `AutoCorrelation`, `FourierBlock`, plain multi-head self-attention) via unused CLI flags; only `Random_Attention_Mechanism=True` is implemented.
- Official RAM sampling and embedding/reduce dropout are hard-coded constants (`connection_probability=0.85`, dropout `0.1`); here they are `spec.py` parameters with the same defaults.
- The reduced representation's trailing axes are `[patch, d_model]` versus the official `[d_model, patch]` (both permute paths cancel to the same patch-embedding step; see the `model.py` docstring). The head is flatten-then-linear, so the function class is unchanged.
- Reference recipes (data pipeline, optimizer schedule, checkpoint and metric comparison against the paper) are out of scope.

## Component decisions

No new shared component was extracted. RAM and the finest-then-pairwise MPMC are this paper's novelty with no other catalog consumer (curate-components requires two). The per-scale linear patch embedding stays local because it differs from `embed.PatchEmbedding` in its bias term and its learnable (not fixed sinusoidal) position table.

## Citation

```bibtex
@article{zhang2026semixer,
  title={SEMixer: Semantics Enhanced MLP-Mixer for Multiscale Mixing and Long-term Time Series Forecasting},
  author={Zhang, Xu and Wang, Qitong and Wang, Peng and Wang, Wei},
  journal={arXiv preprint arXiv:2602.16220},
  year={2026}
}
```
