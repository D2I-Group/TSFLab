# TiRex — reference

## Paper

TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning (NeurIPS 2025, arXiv 2505.23719).

Zero-shot forecasting via in-context learning mostly relies on Transformers, while LSTMs track state well but lack strong in-context learning. TiRex uses xLSTM, an enhanced LSTM with competitive in-context learning, and keeps state tracking, which the authors call critical for long-horizon forecasting. A training-time masking strategy, CPM, further supports state tracking. The paper reports state-of-the-art zero-shot results on GiftEval and Chronos-ZS against larger models (TabPFN-TS, Chronos Bolt, TimesFM, Moirai).

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2505-23719,
  author     = {Andreas Auer and Patrick Podest and Daniel Klotz and
                Sebastian B{\"{o}}ck and G{\"{u}}nter Klambauer and Sepp Hochreiter},
  title      = {TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced
                In-Context Learning},
  journal    = {CoRR},
  volume     = {abs/2505.23719},
  year       = {2025},
  url        = {https://doi.org/10.48550/arXiv.2505.23719},
  doi        = {10.48550/ARXIV.2505.23719},
  eprinttype = {arXiv},
  eprint     = {2505.23719}
}
```
