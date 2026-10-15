# TimePro — reference

## Paper

TimePro: Efficient Multivariate Long-term Time Series Forecasting with Variable- and Time-Aware
Hyper-state (arXiv:2505.20774, 2025).

TimePro patches each channel (as PatchTST does), concatenates each channel's patch embeddings into one
token per variate, and runs a selective state-space ("Mamba"-style) mixer across the variate axis whose
hidden state is also propagated across a 2-D grid of (time-patch, variate) positions before read-out:
the paper's "hyper-state" (Section 3, Figure 2).

## Differences in detail

- The official code (`xwmaxwma/TimePro` at `70a20e5a257b30eb026ee4316293cf4feeb92a1f`) was read to resolve the patch/hidden-dimension relationship, the bidirectional block wiring, and the exact reshape that turns the scan's hidden state into a `(patch_num, n_var)` grid.
- It depends on `selective_scan_cuda_oflex_rh` (custom CUDA selective-scan kernel) and `DCNv4` (CUDA-only deformable convolution); neither is CPU-portable, so neither is used or imported.
- Selective scan: `hyper_state_scan.diagonal_selective_scan`, a pure-PyTorch sequential recurrence over the same `d_state=1` (scalar-state) formulation the paper uses by default.
- Hyper-state mixing: `DCNv4` samples its receptive field from learned per-position offsets; `hyper_state_scan.GridStateMixer` is a fixed-receptive-field depthwise 2-D convolution over the same `(time-patch, variate)` grid, keeping the "mix a state with its neighbours" semantics without learned offsets.
- `DropPath` is a training-time regularizer, not part of the forecasting computation; it is always identity here.

## Implementation notes

- Patch embedding end-pads each channel by `stride` (replicate) before unfolding.
- Token width is `hidden_dim = d_model * patch_num`; the mixer's inner width `expand * hidden_dim` must be a multiple of `patch_num`.
- `d_conv` is the depthwise convolution width inside each scan.
