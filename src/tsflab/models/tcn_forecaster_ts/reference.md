# TCNForecasterTS — reference

## Paper

An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling (Bai,
Kolter, Koltun; arXiv:1803.01271, 2018).

A systematic evaluation of generic convolutional and recurrent architectures across standard
sequence-modelling benchmarks finds that a simple convolutional architecture (TCN) outperforms
canonical recurrent networks such as LSTMs across diverse tasks and datasets while showing longer
effective memory, so convolutional networks should be a natural starting point for sequence modelling.
Official code: http://github.com/locuslab/TCN (not inspected for this implementation).

## Implementation notes

- Block `i` uses dilation `2^i`; causal convolutions left-pad by `(kernel_size - 1) * dilation` so no future step leaks.
- The first block maps `enc_in` input channels to `d_model`; later blocks keep `d_model`.
- Head: `Linear(d_model, pred_len * enc_in)` on the last encoded time step, reshaped to `[B, pred_len, enc_in]`.
- `use_revin = false` disables normalization.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-1803-01271,
  author  = {Shaojie Bai and J. Zico Kolter and Vladlen Koltun},
  title   = {An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling},
  journal = {CoRR},
  volume  = {abs/1803.01271},
  year    = {2018},
  url     = {http://arxiv.org/abs/1803.01271}
}
```
