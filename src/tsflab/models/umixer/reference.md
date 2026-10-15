# UMixer — reference

## Paper

U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting (AAAI 2024, arXiv 2401.02236).

Non-stationarity from trends, seasonality, and irregular fluctuations obstructs stable feature propagation through deep layers. U-Mixer combines Unet and Mixer to capture local dependencies between patches and channels separately, avoiding the influence of distribution variations among channels, and merges low- and high-level features. Its key contribution is a stationarity correction that constrains the stationarity difference between data before and after processing to restore non-stationary information. It reports 14.5% and 7.7% improvements over state-of-the-art methods.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/Ma0FZ024,
  author    = {Xiang Ma and Xuemei Li and Lexin Fang and Tianlong Zhao and Caiming Zhang},
  editor    = {Michael J. Wooldridge and Jennifer G. Dy and Sriraam Natarajan},
  title     = {U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for
               Time Series Forecasting},
  booktitle = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI}
               2024, Thirty-Sixth Conference on Innovative Applications of Artificial
               Intelligence, {IAAI} 2024, Fourteenth Symposium on Educational Advances
               in Artificial Intelligence, {EAAI} 2014, February 20-27, 2024, Vancouver,
               Canada},
  pages     = {14255--14262},
  publisher = {{AAAI} Press},
  year      = {2024},
  url       = {https://doi.org/10.1609/aaai.v38i13.29337},
  doi       = {10.1609/AAAI.V38I13.29337}
}
```
