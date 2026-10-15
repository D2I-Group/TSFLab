# STWave — reference

## Paper

When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph
Attention Networks (Fang et al., ICDE 2023, pp. 517-529; arXiv:2112.02740).

Existing traffic models capture short-term events and long-term daily trends in a single method,
under-use graph positional encoding, and pay quadratic cost for full graph attention. STWave
decomposes traffic with a discrete wavelet transform into low- and high-frequency components,
encodes them with a dual-channel encoder, and uses a wavelet-based graph positional encoding with
query sampling in spectral graph attention. Four real-world datasets show higher precision at lower
computational cost.

## Implementation notes

- `wavelet_disentangle` stays model-local: no cataloged DWT component exists, `wpmixer`'s generic
  multi-level orthogonal filter bank belongs to a different extraction, and STWave's fixed
  single-level same-length Haar reconstruction (upsampled back to the original length, not a
  half-length subband split) is not provably reducible to that generic contract.
- Odd lookbacks are replicate-padded by one step before the Haar split and cropped back.
- Spectral positions: eigenvectors of the normalized Laplacian of the symmetrized adjacency with self-loops.

## Citation

```bibtex
@inproceedings{DBLP:conf/icde/FangQL0XZ023,
  author    = {Yuchen Fang and Yanjun Qin and Haiyong Luo and Fang Zhao and Bingbing Xu and Liang Zeng and Chenxing Wang},
  title     = {When Spatio-Temporal Meet Wavelets: Disentangled Traffic Forecasting via Efficient Spectral Graph Attention Networks},
  booktitle = {39th {IEEE} International Conference on Data Engineering, {ICDE} 2023},
  pages     = {517--529},
  year      = {2023},
  doi       = {10.1109/ICDE55515.2023.00046}
}
```
