# TimeGS — reference

## Paper

Forecasting as Rendering: A 2D Gaussian Splatting Framework for Time Series Forecasting (Wang, Hu,
Liu, Li, Dai, Xia; KDD 2026; arXiv:2603.02220).

## Implementation mapping

- `Model.fold` and `UNetEncoder` (Eqs. 4-6): left zero-pad, fold into rows of `fold_width`, bilinear upsample to `(image_height, image_width)`, then per branch a UNet-style encoder (stride-2 downsampling, residual bottleneck, transposed-conv upsampling with skip concatenation, tanh output) and a linear map to `G` anchor features.
- `gaussian_basis_bank` (Eqs. 7-8): one truncated Gaussian profile per combination of the discretized Cholesky entries `l11 x l21 x l22` (`cholesky1..3`) and an exponent `coefficient` on a `rows x draft_len` grid, each normalized to unit sum; a non-trainable buffer.
- `Model.branch_kernels` (MB-GKG, Eqs. 9-11): softmax (temperature `0.1`) basis mixing weights and MLP intensities give `P` composite kernels per anchor.
- `render_kernels` (MP-CCR, Eqs. 12-14): crop or zero-pad columns to the branch period, flatten row-major, place the kernel centre on the anchor.
- `Model.aggregation_weights` (Eqs. 15-17): channel-specific branch and component weights.
- Anchors: `G = pred_len // ratio + 2 * extend_len`; anchor `g` of branch `k` sits at `(g - extend_len) * ratio + floor(k * ratio / K)`.

## Differences in detail

- Folding: the official `Model.get_img` folds every branch's input with `self.ori_width = 24` regardless of the branch periods (ETTm1 runs periods 96, Weather 144), and all `K` encoders get the same image; reproduced via `fold_width` (default 24).
- Basis bank: the code centres each row's profile at column `draft_len // 2 - l21 j / l11`, truncates with `|dx| <= sqrt(1 - (l22 j)^2) / l11` instead of `δᵀΣ⁻¹δ ≤ 1`, and multiplies the exponent by an extra `coefficient` axis; the code construction is followed because it produced the reported results.
- Weight normalization: the code divides each row by its own sum before the softmax (undefined if a row sum reaches zero); `weight_norm` selects code or paper behaviour.
- Unstated in the paper and taken from `layers/TimeGS_Enc.py` (`GlobalGenerator`, `ResnetBlock`), `run.py` defaults and `scripts/long_term_forecast/ETTh1.sh`: the UNet layout (stem, three stride-2 stages, residual bottleneck with `Dropout(0.5)`, skip concatenation, tanh output), the basis grid (`rows = 7`, `draft_len = 11`), anchor stride/extension (`ratio`, `extend_len`), and the basis-bank entries.
- Preset (ETTh1, horizon 96): `periods = [24, 24, 24]`, `components = 2`, `n_blocks = 1`, `hidden_dim = 16`, `conv_dim = 8`, 24 x 24 image.
- Rendering is computed per branch on one device, numerically the same as the official batched single-device path.
- Constraints: `image_height` and `image_width` divisible by `2 ** n_downsampling`; odd `kernel_size`.

## Citation

```bibtex
@inproceedings{wang2026timegs,
  title     = {Forecasting as Rendering: A 2D Gaussian Splatting Framework for Time Series Forecasting},
  author    = {Wang, Yixin and Hu, Yifan and Liu, Peiyuan and Li, Naiqi and Dai, Tao and Xia, Shu-Tao},
  booktitle = {Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)},
  year      = {2026},
  note      = {arXiv:2603.02220}
}
```
