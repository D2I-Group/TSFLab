# BigST — reference

## Paper

- **Title**: BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks
- **Venue**: Proceedings of the VLDB Endowment (PVLDB), Vol. 17, No. 5, pp. 1081-1090, 2024

## Abstract

Most spatio-temporal GNNs need quadratic complexity to capture long-range dependencies, which blocks long histories on large road networks.
BigST pre-computes low-dimensional node representations from long inputs (thousands of past steps) with a scalable long-sequence feature extractor, and builds a linearized global spatial convolution that distils time-varying graph structure with linear-complexity message passing.
On two large traffic datasets it scales to road networks of up to one hundred thousand nodes while improving accuracy and efficiency.

## Citation

```bibtex
@article{han2024bigst,
  author  = {Jindong Han and Weijia Zhang and Hao Liu and Tao Tao and Naiqiang Tan and Hui Xiong},
  title   = {BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks},
  journal = {Proc. VLDB Endow.},
  volume  = {17},
  number  = {5},
  pages   = {1081--1090},
  year    = {2024},
  doi     = {10.14778/3641204.3641217}
}
```
