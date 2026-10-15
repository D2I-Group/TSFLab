# iTransformer — reference

## Paper

Liu et al., ICLR 2024 (arXiv 2310.06625, https://arxiv.org/abs/2310.06625).

Transformer forecasters that tokenize timestamps degrade and grow costly with longer lookbacks, and each temporal
token fuses variates with different physical meaning, giving meaningless attention maps. iTransformer applies
attention and the feed-forward network on the inverted dimensions without modifying the components: each
variate's time points form a variate token, attention captures multivariate correlation, and the FFN learns
per-variate representations. It reaches state of the art and generalizes across variates and lookback lengths.

## Citation

```bibtex
@inproceedings{DBLP:conf/iclr/LiuHZWWML24,
  author    = {Yong Liu and Tengge Hu and Haoran Zhang and Haixu Wu and Shiyu Wang and
               Lintao Ma and Mingsheng Long},
  title     = {iTransformer: Inverted Transformers Are Effective for Time Series Forecasting},
  booktitle = {The Twelfth International Conference on Learning Representations, {ICLR} 2024},
  publisher = {OpenReview.net},
  year      = {2024},
  url       = {https://openreview.net/forum?id=JePfAI8fah}
}
```
