# MTGNN — reference

## Paper

- **Title**: Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks
- **Venue**: KDD 2020 (arXiv 2005.11650, 2020-05)
- **Abstract (shortened)**: Multivariate forecasting assumes variables depend on one another, but GNNs need a well-defined graph that multivariate series usually lack. MTGNN learns uni-directed relations among variables with a graph learning module (into which external knowledge such as variable attributes can be integrated), and captures spatial and temporal dependence with a mix-hop propagation layer and a dilated inception layer, all trained end to end. It outperformed baselines on 3 of 4 benchmark datasets and was on par on two traffic datasets that provide extra structural information.

## Implementation notes

- Learned graph: `relu(tanh(alpha (M1 M2^T - M2 M1^T)))` with `M_i = tanh(alpha W_i E_i)`, top-k per row, then row-normalized; it is blended with the self-loop-augmented, row-normalized predefined graph by `sigmoid(graph_mix)` and renormalized.
- Each layer sums mix-hop propagation over the graph and over its transpose, then applies a residual linear map, dropout, and LayerNorm.
- The input projection reads `input_dim` features (value plus marks); missing feature channels are zero-padded and extra ones dropped.

## Citation

```bibtex
@inproceedings{DBLP:conf/kdd/WuPL0CZ20,
  author    = {Zonghan Wu and Shirui Pan and Guodong Long and Jing Jiang and Xiaojun Chang and Chengqi Zhang},
  title     = {Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks},
  booktitle = {{KDD} '20: The 26th {ACM} {SIGKDD} Conference on Knowledge Discovery and Data Mining},
  pages     = {753--763},
  publisher = {{ACM}},
  year      = {2020},
  doi       = {10.1145/3394486.3403118}
}
```
