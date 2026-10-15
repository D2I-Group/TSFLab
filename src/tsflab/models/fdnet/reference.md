# FDNet — reference

## Differences in detail

- Sources read: `FDNet/model.py`, `FDNet/ConvBlock.py`, `FDNet/embed.py`,
  `exp/exp_model.py`, `scripts/FDNet.sh`.
- Resolved from the official code (Sec. 5.1-5.2 leave these open):
  `focal_parts` f equals the official `--pyramid` plus one (scripts: pyramid 4,
  f = 5, with 672 inputs; pyramid 0 with 96 inputs); the sub-sequence
  boundaries; dropout after the value embedding and after every convolution
  before its GELU; each residual connection spans one 1x1 + 3x1 pair (two per
  layer); the horizon projection is one linear map over flattened
  `[time, d_model]` features shared by all variates; branch forecasts are
  averaged; defaults `d_model = 8`, `num_layers = 5`, kernel 3, dropout 0.1; no
  instance normalization (data are globally z-scored by the loader).
- Depth schedule: `focal_depths` gives N, N, N-1, ..., N-f+2.
- The official model builds one more branch than it has inputs; that branch
  never receives data.
- The paper lists MSE as the training loss; the official loop minimizes
  MSE + MAE.
- Checked: focal lengths, split order and depth schedule against the official
  settings, the weight-normalized residual conv pair and four-convolution layer,
  variate independence (perturbing one variate leaves others unchanged), the
  per-variate head, branch averaging, input-length validation. Reported
  benchmark numbers are not reproduction claims of this implementation.

## Citation

Shen, L., Wei, Y., Wang, Y., Qiu, H. "FDNet: Focal Decomposed Network for Efficient, Robust and Practical Time Series Forecasting." Knowledge-Based Systems 275, 110666 (2023). doi:10.1016/j.knosys.2023.110666.
