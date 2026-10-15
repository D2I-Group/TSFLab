# BiST — reference

## Paper

- **Title**: BiST: A Lightweight and Efficient Bi-Directional Model for Spatiotemporal Prediction
- **Venue**: Proceedings of the VLDB Endowment (PVLDB), Vol. 18, No. 6, pp. 1663-1676, 2025

## Abstract

Spatiotemporal models often assume input-label spatiotemporal consistency, and their complexity limits scalability.
BiST splits prediction into a forward spatiotemporal learning process that generates base predictions and a residual correction process that models spatiotemporal residuals to refine them.
The backbone is a lightweight MLP instead of stacked spatiotemporal layers, giving competitive accuracy at a small fraction of the training time and memory of state-of-the-art models.

## Citation

```bibtex
@article{ma2025bist,
  author  = {Jiaming Ma and Binwu Wang and Pengkun Wang and Zhengyang Zhou and Xu Wang and Yang Wang},
  title   = {BiST: A Lightweight and Efficient Bi-directional Model for Spatiotemporal Prediction},
  journal = {Proc. VLDB Endow.},
  volume  = {18},
  number  = {6},
  pages   = {1663--1676},
  year    = {2025},
  doi     = {10.14778/3725688.3725697}
}
```
