# Sumba — reference

## Paper

Structured Matrix Basis for Multivariate Time Series Forecasting with Interpretable Dynamics
(Chen, Li, Chen, Li; NeurIPS 2024).

Multivariate dynamics combine temporal dependencies with spatial correlations that are structured
and evolve. Two-stage approaches (learn series representations, then generate structures) are
sensitive to small input windows and have high variance. Sumba generates dynamic spatial structures
from a structured matrix basis with structure regularization, giving a well-constrained, lower-variance
yet expressive output space with interpretable dynamics. Six benchmarks show up to 8.5% improvement.

## Implementation notes

- Each basis matrix is a row softmax of a low-rank product `left * diag(spectrum) * right^T` (`basis_rank`), so every basis is row-stochastic.
- Mixing weights come from a small MLP on the hidden state averaged over time and channels, followed by softmax over `basis_count`.
- `diversity_penalty()` returns the mean squared off-diagonal of the normalized basis Gram matrix (orthogonality-style regularizer).
- Normalization: lookback mean and standard deviation per channel, both detached.

## Citation

```bibtex
@inproceedings{DBLP:conf/nips/ChenL0024,
  author    = {Xiaodan Chen and Xiucheng Li and Xinyang Chen and Zhijun Li},
  title     = {Structured Matrix Basis for Multivariate Time Series Forecasting with Interpretable Dynamics},
  booktitle = {Advances in Neural Information Processing Systems 37, NeurIPS 2024},
  year      = {2024},
  url       = {http://papers.nips.cc/paper\_files/paper/2024/hash/2b47305e1c81890b1089a405686ad183-Abstract-Conference.html}
}
```
