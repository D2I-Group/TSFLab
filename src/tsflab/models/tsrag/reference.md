# TSRAG — reference

## Paper

TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster (NeurIPS 2025, arXiv 2503.07649).

Fine-tuned LLMs generalize poorly to unseen datasets, and time series foundation models (TSFMs) lack mechanisms to adapt to non-stationary dynamics and distribution shift. TS-RAG uses pretrained time series encoders to retrieve semantically relevant segments from a dedicated knowledge base and an Adaptive Retrieval Mixer (ARM) to fuse them with the TSFM's internal representation, without task-specific fine-tuning. It reports state-of-the-art zero-shot results on seven benchmarks, up to 6.84% better than existing TSFMs, with added interpretability.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2503-07649,
  author     = {Kanghui Ning and Zijie Pan and Yu Liu and Yushan Jiang and
                James Y. Zhang and Kashif Rasul and Anderson Schneider and
                Lintao Ma and Yuriy Nevmyvaka and Dongjin Song},
  title      = {{TS-RAG:} Retrieval-Augmented Generation based Time Series Foundation
                Models are Stronger Zero-Shot Forecaster},
  journal    = {CoRR},
  volume     = {abs/2503.07649},
  year       = {2025},
  url        = {https://doi.org/10.48550/arXiv.2503.07649},
  doi        = {10.48550/ARXIV.2503.07649},
  eprinttype = {arXiv},
  eprint     = {2503.07649}
}
```
