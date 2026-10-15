# DFDGCN — reference

## Paper

- **Title**: Dynamic Frequency Domain Graph Convolutional Network for Traffic Forecasting
- **Venue**: ICASSP 2024
- **Published**: 2024 (arXiv: 2023-12)
- **arXiv**: https://arxiv.org/abs/2312.11933

## Abstract

Data-driven learning of spatial dependencies in transportation networks is hindered by time-shift of traffic
patterns and by noise. DFDGCN mitigates time-shift with a Fourier transform and adds sensor identity and time
embeddings when learning the graph, combines the learned graph with static predefined and self-adaptive graphs
in graph convolution, and predicts with classical causal convolutions; it beats baselines on four datasets.

## Citation

```bibtex
@inproceedings{DBLP:conf/icassp/LiSXQCW24,
  author    = {Yujie Li and Zezhi Shao and Yongjun Xu and Qiang Qiu and Zhaogang Cao and Fei Wang},
  title     = {Dynamic Frequency Domain Graph Convolutional Network for Traffic Forecasting},
  booktitle = {{IEEE} International Conference on Acoustics, Speech and Signal Processing, {ICASSP} 2024, Seoul, Republic of Korea, April 14-19, 2024},
  pages     = {5245--5249},
  publisher = {{IEEE}},
  year      = {2024},
  url       = {https://doi.org/10.1109/ICASSP48485.2024.10446144},
  doi       = {10.1109/ICASSP48485.2024.10446144}
}
```
