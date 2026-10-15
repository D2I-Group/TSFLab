# LSTM — reference

## Paper

Hochreiter and Schmidhuber, "Long Short-Term Memory", Neural Computation 9(8), 1997.

Recurrent backpropagation learns long time lags very slowly because error backflow decays. LSTM enforces
constant error flow through constant error carousels in special units, with multiplicative gates that learn
to open and close access to it, bridging lags beyond 1000 steps at O(1) cost per time step and weight. It
learns faster and solves long-time-lag tasks earlier recurrent methods could not.

## Citation

```bibtex
@article{hochreiter1997long,
  author  = {Sepp Hochreiter and J{\"u}rgen Schmidhuber},
  title   = {Long Short-Term Memory},
  journal = {Neural Computation},
  volume  = {9},
  number  = {8},
  pages   = {1735--1780},
  year    = {1997},
  doi     = {10.1162/neco.1997.9.8.1735},
  url     = {https://doi.org/10.1162/neco.1997.9.8.1735}
}
```
