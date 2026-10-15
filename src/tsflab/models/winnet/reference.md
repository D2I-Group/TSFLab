# WinNet — reference

## Paper

WinNet: Make Only One Convolutional Layer Effective for Time Series Forecasting (arXiv 2311.00214, 2023; revised 2024-06). Sources used: Section 3 (Eqs. 1-5, Figs. 2-4).

```bibtex
@article{ou2023winnet,
  title   = {WinNet: Make Only One Convolutional Layer Effective for Time Series Forecasting},
  author  = {Ou, Wenjie and Zhao, Zhishuo and Guo, Dongyue and Zhang, Zheng and Lin, Yi},
  journal = {arXiv preprint arXiv:2311.00214},
  year    = {2023}
}
```

## Implementation mapping

Independent rewrite after reading the pinned official code (`ouwen18/WinNet` at `2012c30d`; no license file, recorded as `NOASSERTION`): `WinNet.py` and `run_longExp.py` (identical copies under `Code_of_WinNet/`). Nothing was copied or imported.

Resolved from the official code:

- The input linear maps `seq_len` to `n_times * time_len` points (default 24 x 24 = 576; square because both views are summed). Here the side is the single `window` parameter.
- The 2D moving average is a 24 x 24 average pool over a tensor padded by 12 before and 11 after.
- The convolution is `Conv2d(2, 1, 3)` with bias, one per channel, shared by the two heads; dropout 0.5 follows the sigmoid.
- The sub-window tensor itself is added to the two head outputs; the cross-window head output is summed in its transposed layout.
- The official long-term setting uses `seq_len = 512`, MSE loss, learning rate `1e-3`.
