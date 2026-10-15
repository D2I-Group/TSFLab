# LightTS — reference

## Paper

Zhang et al., "Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP
Structures", arXiv 2207.01186 (2022).

Complex RNN, GNN, and Transformer forecasters are expensive on large data. LightTS applies simple MLP
structures on top of two down-sampling strategies, interval and continuous sampling, because down-sampled
series retain most of their information. On eight benchmarks it is best on five and comparable on the rest,
uses under 5% of the FLOPs of previous SOTA on the largest dataset, and has lower variance on long horizons.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2207-01186,
  author  = {Tianping Zhang and Yizhuo Zhang and Wei Cao and Jiang Bian and Xiaohan Yi and
             Shun Zheng and Jian Li},
  title   = {Less Is More: Fast Multivariate Time Series Forecasting with Light
             Sampling-oriented {MLP} Structures},
  journal = {CoRR},
  volume  = {abs/2207.01186},
  year    = {2022},
  doi     = {10.48550/ARXIV.2207.01186}
}
```
