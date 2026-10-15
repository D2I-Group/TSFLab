# TimeXer — reference

## Paper

TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables (NeurIPS 2024, arXiv 2402.19072).

Real systems are partially observed, so forecasting only the endogenous target is often insufficient; exogenous variables carry useful external information. TimeXer adapts the canonical Transformer with patch-wise self-attention over the endogenous series and variate-wise cross-attention to exogenous series, and learns global endogenous tokens that bridge exogenous information into the endogenous patches. The paper reports state-of-the-art results on twelve real-world benchmarks.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/WangWDQZLQWL24,
  author    = {Yuxuan Wang and Haixu Wu and Jiaxiang Dong and Guo Qin and
               Haoran Zhang and Yong Liu and Yunzhong Qiu and Jianmin Wang and
               Mingsheng Long},
  editor    = {Amir Globersons and Lester Mackey and Danielle Belgrave and
               Angela Fan and Ulrich Paquet and Jakub M. Tomczak and Cheng Zhang},
  title     = {TimeXer: Empowering Transformers for Time Series Forecasting with
               Exogenous Variables},
  booktitle = {Advances in Neural Information Processing Systems 37: Annual Conference
               on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver,
               BC, Canada, December 10 - 15, 2024},
  year      = {2024},
  url       = {http://papers.nips.cc/paper\_files/paper/2024/hash/0113ef4642264adc2e6924a3cbbdf532-Abstract-Conference.html}
}
```
