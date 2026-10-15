# ModernTCN — reference

## Paper

- **Title**: ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis
- **Venue**: ICLR 2024
- **Abstract (shortened)**: Convolution had lost ground to Transformer and MLP models in time series analysis. The paper modernizes the traditional TCN with time-series-specific modifications. As a pure convolution structure, ModernTCN reaches consistent state-of-the-art results on five mainstream analysis tasks while keeping the efficiency of convolution models. Compared with earlier convolution models it has much larger effective receptive fields.

## Implementation mapping

- The backbone preserves the variable axis: `[batch, variables, width, patches]`.
- Stage 0 embeds each variable with an unpadded `Conv1d(1, width, patch_size, stride=patch_stride)`; later stages downsample with `Conv1d(kernel=stride=downsample_ratio)`.
- Each block: large-kernel depthwise convolution (with optional small-kernel branch), BatchNorm, variable-grouped ConvFFN, then feature-grouped ConvFFN, with a residual connection.
- Stage lists `num_blocks`, `large_size`, `small_size`, `dims` must have equal length; all temporal kernels must be odd.
- The head flattens the last stage (or all stages with `use_multi_scale`) and maps linearly to `pred_len`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/LuoW24,
  author    = {Donghao Luo and Xue Wang},
  title     = {ModernTCN: {A} Modern Pure Convolution Structure for General Time Series Analysis},
  booktitle = {The Twelfth International Conference on Learning Representations, {ICLR} 2024},
  publisher = {OpenReview.net},
  year      = {2024},
  url       = {https://openreview.net/forum?id=vpJMJerXHU}
}
```
