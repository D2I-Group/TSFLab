# TimesNet — reference

## Paper

TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis (Wu, Hu, Liu, Zhou, Wang,
Long; ICLR 2023; arXiv:2210.02186).

Temporal variations are hard to model directly in 1D. Based on multi-periodicity, TimesNet unravels
them into intra-period and inter-period variations by transforming the 1D series into a set of 2D
tensors based on multiple periods, embedding the two kinds of variation into columns and rows so that
2D kernels can model them. TimesBlock discovers the periods adaptively and extracts variations with a
parameter-efficient inception block. TimesNet reports consistent state of the art on short- and
long-term forecasting, imputation, classification and anomaly detection.

## Implementation notes

- Inputs `[B, seq_len, enc_in]` with active six-column marks; outputs `[B, pred_len, c_out]`.
- Each TimesBlock pads the hidden sequence to a multiple of each detected period before folding and crops it back afterwards.
- Aggregation weights are the softmax over the selected periods' FFT amplitudes, per sample.
- `e_layers` TimesBlocks, each followed by LayerNorm; a linear projection maps `d_model` to `c_out`.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/WuHLZ0L23,
  author    = {Haixu Wu and Tengge Hu and Yong Liu and Hang Zhou and Jianmin Wang and Mingsheng Long},
  title     = {TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  booktitle = {The Eleventh International Conference on Learning Representations, {ICLR} 2023},
  publisher = {OpenReview.net},
  year      = {2023},
  url       = {https://arxiv.org/abs/2210.02186}
}
```
