# GCLSTM — reference

## Paper

- **Title**: A hybrid model for spatiotemporal forecasting of PM2.5 based on graph convolutional neural network and long short-term memory
- **Venue**: Science of the Total Environment, vol. 664, pp. 1-10 (2019)

## Abstract (shortened)

GC-LSTM models historical observations at different stations as spatiotemporal graph series, with air quality variables, meteorological factors, and temporal attributes as graph signals.
Graph convolutions extract spatial dependency between stations and an LSTM captures temporal dependency.
On real-world data it reported a recall of 68.45% and a false alarm rate of 4.65% (threshold 115 μg/m³) and R² of 0.72 for 72-hour PM2.5 forecasts, and the authors suggest it applies to other pollutants.

## Citation

```bibtex
@article{Qi2019GCLSTM,
  author    = {Yanlin Qi and Qi Li and Hamed Karimian and Di Liu},
  title     = {A hybrid model for spatiotemporal forecasting of PM2.5 based on graph
               convolutional neural network and long short-term memory},
  journal   = {Science of The Total Environment},
  volume    = {664},
  pages     = {1--10},
  year      = {2019},
  doi       = {10.1016/J.SCITOTENV.2019.01.333}
}
```
