# GRAMODE — reference

## Differences in detail

- Paper source: arXiv 2305.18687 (TMLR version; Sec. 3-5, Eqs. 1-24, Algorithms 1-2, Sec. 5.4).
- Implementation: independent rewrite after reading the pinned official code (`zbliu98/GRAM-ODE`, revision `aa4e2dada12d9dd6379a10ce3bef39b62972040c`, no license file): `model.py`, `odegcn.py`, `utils.py`, `run_stode.py`, `args.py` and `eval.py`. Nothing was copied or imported.
- Resolved from the official code: three TCN levels of widths `(64, 32, 64)`, kernel 2 and causal dilations 1, 2, 4, with weights drawn from `N(0, 0.01^2)`. ODE width is 64. Each graph ODE takes one Euler step (`t = [0, 1]`), and each local ODE takes `T / K = 3` steps (`t = [0, 1, 2, 3]`). The shared spatial mask `M`, the temporal gate weights, `e` and the update weights are drawn from a standard normal. The per-node `alpha` starts at `0.8` and `W` at the identity. The local attention maps `T * 64` to `K` tokens with key width `8 * 64` and 32 heads, and the output attention has key width `6 * 64` and 12 heads. `BatchNorm2d` treats nodes as channels. Graph normalization uses `alpha = 0.8`. The loss is `SmoothL1Loss`, with AdamW at learning rate `2e-4`, milestones `[15, 50, 105, 145]` and decay `0.13`.
- The forecast is the model output in the runner's scale; the official pipeline de-normalizes inside the training loop.
- The official PEMS04 and PEMS08 runs use flow, occupancy and speed, which the catalog datasets do not expose.
- The DTW graph follows the official transform (kept as `semantic_sigma`, `semantic_threshold`) because the published results depend on it, though it differs from Eq. (2); `fastdtw` (radius 6) is replaced by exact DTW with the absolute-difference cost.

## Verification

Checked properties: the Eq. (3) normalization with an isolated node; the day profiles; the vectorized DTW against a naive dynamic program; the official DTW-graph transform; the node and edge temporal gates against the official tensor layout; the node and edge ODE derivatives; the Euler steps with a detached initial state; TCN causality and the always-active residual; the official edge-feature construction; the message filter; the aggregation; the segment order of the local messages; forward shapes and gradients through every block; the Huber objective; DTW graph fitting from an index-windowed training split (and the warning when no series is available). Reported benchmark numbers are not reproduction claims.

## Citation

```bibtex
@article{liu2023gramode,
  title   = {Graph-based Multi-{ODE} Neural Networks for Spatio-Temporal Traffic Forecasting},
  author  = {Liu, Zibo and Shojaee, Parshin and Reddy, Chandan K.},
  journal = {Transactions on Machine Learning Research},
  issn    = {2835-8856},
  year    = {2023},
  url     = {https://openreview.net/forum?id=Oq5XKRVYpQ}
}
```
