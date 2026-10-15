# STPGNN — reference

## Paper

Spatio-Temporal Pivotal Graph Neural Networks for Traffic Flow Forecasting (Kong, Guo, Liu; AAAI 2024, pp. 8627-8635).

Most GNN traffic forecasters ignore pivotal nodes, which have extensive connections to many other
nodes and complex spatio-temporal dependencies. STPGNN adds a pivotal node identification module, a
pivotal graph convolution centred on pivotal nodes, and a parallel framework extracting features on
pivotal and non-pivotal nodes. Experiments on seven real-world traffic datasets report gains in
accuracy and efficiency over state-of-the-art baselines.

## Implementation notes

- Node scores = learned degree of `sigmoid(source @ target.T)` plus physical degree of the normalized adjacency.
- The adaptive affinity is row-normalized and added to the physical graph before every forward pass.
- Without `adj_mx`, the physical graph is the identity (after self-loops and normalization).
- Readout flattens `seq_len * residual_channels` per node into an MLP of width `end_channels`.

## Citation

```bibtex
@inproceedings{DBLP:conf/aaai/KongGL24,
  author    = {Weiyang Kong and Ziyu Guo and Yubao Liu},
  title     = {Spatio-Temporal Pivotal Graph Neural Networks for Traffic Flow Forecasting},
  booktitle = {Thirty-Eighth {AAAI} Conference on Artificial Intelligence, {AAAI} 2024},
  pages     = {8627--8635},
  publisher = {{AAAI} Press},
  year      = {2024},
  doi       = {10.1609/AAAI.V38I8.28707}
}
```
