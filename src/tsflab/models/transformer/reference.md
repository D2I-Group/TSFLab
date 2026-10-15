# Transformer — reference

## Paper

Attention Is All You Need (NeurIPS 2017, arXiv 1706.03762).

Proposes the Transformer, a sequence transduction architecture based solely on attention, without recurrence or convolution, connecting encoder and decoder through attention. It was more parallelizable and faster to train than recurrent or convolutional encoder-decoders, reaching 28.4 BLEU on WMT 2014 English-to-German and 41.8 BLEU on English-to-French, and generalized to English constituency parsing.

## Citation

```bibtex
@misc{vaswani2017attention,
  author        = {Ashish Vaswani and Noam Shazeer and Niki Parmar and
                   Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and
                   Lukasz Kaiser and Illia Polosukhin},
  title         = {Attention Is All You Need},
  year          = {2017},
  eprint        = {1706.03762},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/1706.03762}
}
```
