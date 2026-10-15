# PPGF — reference

## Differences in detail

- Paper source: arXiv 2502.12802 (Sec. III-A to III-F, Eqs. 1-11, Sec. IV-B).
- Implementation: independent rewrite after reading the pinned official code (`syrGitHub/PPGF`, revision `a478f95f26d9f76da4ea5264e1b7bb9473360176`, no license file): `model_multi_task.py` (`ModelB`), `layer/mytransformerencoder.py`, `layer/GRNlayer.py`, `Group_helper.py`, `train.py`, `test.py`, `data.py`, and `main.py`. Nothing was copied or imported.
- Resolved from the official code: `Conv1d(1, 32, kernel 2)` without padding, a fixed sinusoidal position table with dropout 0.3, and one encoder layer (8 heads, key and value width 64, bias-free projections, scaling by `sqrt(64)`, post-norm residuals, feed-forward `32 -> 2048 -> 32` with ReLU and no bias).
- Gated residual networks: `fc1 -> fc2` without activation, Kaiming-normal and zero-bias initialization, a dropout and Xavier-initialized GLU, and an add-and-LayerNorm; a residual of another width is linearly interpolated, gated by `2 sigmoid(m)` and normalized.
- Variable selection: one GRN per position and softmax weights from a GRN on the flattened positions, with hidden width 416 and dropout 0.2.
- Heads: `FC1 = Linear(416, hidden)`, an auxiliary TCP classifier, a confidence `Linear(hidden, 1)` without squashing, `FC2` on `h * c_hat`, and one ReLU offset per group.
- Groups: the non-symmetric `Group_helper` rule (order statistics at `floor((t - 1) k / K)`); labels are open-ended in the first and last group; a degenerate group adds the raw offset to its edge.
- Per-channel, per-step heads: every channel is forecast independently and every horizon step has its own classifier, confidence and offset block, where the code predicts one scalar of one channel.
- Group edges are fitted on the whole scaled training series of each channel (the paper's `A`), where the code uses the training targets. Before `training_setup` runs, the edges default to standard-normal quantiles with ends at -3 and 3.
- Objective: the TCP target is detached (as in ConfidNet). Loss weights are the paper's `(1, 1, 5)`, and training uses the catalog optimizer and schedule (the paper trains with Adam, batch 32, weight decay 1e-4). The unused `AR`, focal-loss and two-step variants are not implemented.
- Limits: offsets are only bounded below, so a forecast can exceed its group's upper edge. Values outside the training range fall in the first or last group with an offset outside [0, 1].
- Checked behaviour: group edges of Eqs. (1)-(2) and labels against a re-statement of the official `Group_helper` rules, degenerate-group handling and the label-to-forecast round trip of Eq. (10), GRN, variable-selection and encoder-layer outputs against direct references, the calibrated classifier of Eq. (6), forecast placement, channel independence, edge fitting (series and fallback), the objective of Eq. (11) with its per-group regression term for `M` and `MS` targets, gradients, and the strict parameter schema. Reported benchmark numbers are not reproduction claims of this implementation.

## Paper

PPGF: Probability Pattern-Guided Time Series Forecasting, IEEE Transactions on Neural Networks and Learning Systems, 2025 (arXiv 2502.12802).

## Citation

```bibtex
@article{sun2025ppgf,
  title   = {PPGF: Probability Pattern-Guided Time Series Forecasting},
  author  = {Sun, Yanru and Xie, Zongxia and Xing, Haoyu and Yu, Hualong and Hu, Qinghua},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year    = {2025}
}
```
