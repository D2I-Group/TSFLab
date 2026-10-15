# PaiFilter — reference

## Paper

- **Title**: FilterNet: Harnessing Frequency Filters for Time Series Forecasting
- **Venue**: NeurIPS 2024 (arXiv 2411.01623, 2024-11)
- **Abstract (shortened)**: Transformer forecasters remain vulnerable to high-frequency signals, costly to compute, and limited in full-spectrum use. FilterNet borrows from signal processing: learnable frequency filters selectively pass or attenuate components of the series. Two filters are proposed: a plain shaping filter (a universal frequency kernel, implemented here) and a contextual shaping filter that conditions the filter on the input. The paper argues the filters approximately surrogate linear and attention mappings, and reports gains in accuracy and efficiency on eight benchmarks.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/0001FZHHL024,
  author    = {Kun Yi and Jingru Fei and Qi Zhang and Hui He and Shufeng Hao and Defu Lian and Wei Fan},
  editor    = {Amir Globersons and Lester Mackey and Danielle Belgrave and Angela Fan and
               Ulrich Paquet and Jakub M. Tomczak and Cheng Zhang},
  title     = {FilterNet: Harnessing Frequency Filters for Time Series Forecasting},
  booktitle = {Advances in Neural Information Processing Systems 37: Annual Conference
               on Neural Information Processing Systems 2024, NeurIPS 2024},
  year      = {2024},
  url       = {http://papers.nips.cc/paper\_files/paper/2024/hash/6323d96f79d5d49e0d3fc88835c082cd-Abstract-Conference.html}
}
```
