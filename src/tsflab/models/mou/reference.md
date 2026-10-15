# MoU — reference

## Paper

- **Title (arXiv 2408.15997)**: Mamba or Transformer for Time Series Forecasting? Mixture of Universals (MoU) Is All You Need
- **Camera-ready title (KDD 2025)**: Semantics-Aware Patch Encoding and Hierarchical Dependency Modeling for Long-Term Time Series Forecasting

## Differences in detail

Paper and official code (pinned revision `5a73fd7`: `layers/MoU_backbone.py`,
`layers/MoE3.py`, `models/MoU.py`) were checked; the local module is an
independent rewrite using the `entype=mof`, `ltencoder=mfca` configuration.

- The official `Encoder_MFCA.forward` assigns `output = block(x)` inside its loop, so only the last block's output is used and earlier blocks are dead for `e_layers > 1`. The local `blocks` are stacked sequentially.
- The Mamba mixer is the pure-PyTorch `mamba` component with `d_conv=4`, `expand=2`, `dt_rank=ceil(d_model/16)`, and `d_state=21`.
- The self-attention layer is `nn.MultiheadAttention` in a post-norm (BatchNorm) encoder layer. The official layer is the PatchTST `TSTEncoderLayer` with residual attention scores; with one attention layer per block the two are equivalent up to parameterization.
- The official sparse MoE (`MoE3.py`) dispatches tokens per expert and its load-balancing loss is commented out; the local gate computes all linear extractors densely and zeroes non-top-k weights (top-k of the softmax, renormalized with eps 1e-6, noise `softplus + 1e-2` in training).
- RevIN uses `affine=False` (official default `--affine 0`); end replication padding and patch count `(seq_len - patch_len) // stride + 2` match the official `padding_patch='end'`. The official series-decomposition option and the `se`/`dyconv`/`w` entype and alternative long-term encoder variants are not implemented.
- Dropouts hard-coded in the official block (Mamba 0.1, convolution 0.3) are exposed as `mamba_dropout` and `conv_dropout`; the rest follow the official `--dps` order (`ffn`, `block`, `attn`, `tf`).
- Preset sizes (`d_model=128`, `n_heads=16`, `d_ff=256`, `e_layers=1`) are TSFLab defaults; `d_state=21`, `num_experts=4`, `top_k=2` match the official defaults, while the official ETTh1 script uses `d_model=64`, `n_heads=4`, `d_ff=128`, `seq_len=336`.

## Citation

```bibtex
@inproceedings{peng2025semantics,
  title     = {Semantics-Aware Patch Encoding and Hierarchical Dependency Modeling for Long-Term Time Series Forecasting},
  author    = {Peng, Sijia and Xiong, Yun and Zhu, Yangyong and Shen, Zhiqiang},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  pages     = {2269--2280},
  year      = {2025},
  doi       = {10.1145/3711896.3737123}
}
```
