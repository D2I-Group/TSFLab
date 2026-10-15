# SVRForecasterTS — reference

## Paper

Support Vector Regression Machines (Drucker, Burges, Kaufman, Smola, Vapnik; NIPS 1996, pp. 155-161).

Support Vector Regression uses an epsilon-insensitive objective and a kernel expansion. The local
adaptation keeps those two ideas but learns RBF centres and coefficients by gradient descent instead
of solving the constrained dual problem.

## Implementation notes

- Features: `exp(-kernel_gamma * ||x - c_j||^2)` for each of `num_support` centres `c_j` of length `seq_len`.
- Forecast: `features @ coefficients + bias`, with `coefficients` of shape `[num_support, pred_len]`.
- `aux_loss = 0.5 * l2_penalty * ||coefficients||^2`, recomputed on each forward pass.
- `epsilon_insensitive_loss(pred, target) = mean(max(|pred - target| - epsilon, 0))`.
- No input normalization is applied inside the model.

## Citation

```bibtex
@inproceedings{drucker1996support,
  author    = {Harris Drucker and Christopher J. C. Burges and Linda Kaufman and Alexander J. Smola and Vladimir Vapnik},
  title     = {Support Vector Regression Machines},
  booktitle = {Advances in Neural Information Processing Systems 9 (NIPS 1996)},
  pages     = {155--161},
  year      = {1996},
  url       = {https://proceedings.neurips.cc/paper/1996/hash/d38901788c533e8286cb6400b40b386d-Abstract.html}
}
```
