# HADL — reference

## Differences in detail

- Implementation: independent rewrite of Sec. 3 (Eqs. 1-5) and Algorithm 1 of arXiv 2502.10569. The official repository (`forgee-master/HADL`, revision `aecbda0dafd7c1ab6bcdb2d3b0e1f0f053f8e30b`, `LICENSE` Apache-2.0) was read for `models/HADL.py`, `exp/exp_main.py`, `run_longExp.py` and `scripts/multivariate/HADL/*.sh`; nothing was copied or imported.
- Resolved from the official code: per-channel window-mean centering restored after the layer; Haar low-pass filter `[1, 1]/sqrt(2)` with stride 2 and zero padding of odd lengths; scipy orthonormal DCT-II divided by the transformed length; low-rank factors `A: [in, rank]`, `B: [rank, out]` Kaiming-uniform initialized as stored, bias `U(0, 1)`; one layer shared by all channels (`--individual 0`); L1 penalty `0.1 * mean|forecast|` added to the MSE loss.
- Script settings: `seq_len = 512`, rank 50, learning rate 0.01, batch size 32, 100 epochs with patience 20.
- Shared components, checked against the reference: `haar_dwt1d.HaarDWT1D` computes the same orthonormal approximation for even lengths; the model zero-pads odd lengths itself before calling it (the component would replicate the last sample), so results equal the official low-pass convolution for every length.
- The DCT is a fixed local matrix (only this model uses it); it replaces the official CPU round trip through scipy with an equivalent differentiable matrix product, whose transpose is the exact orthonormal inverse used by the official backward.
- The official inverse-DCT forward uses the unnormalized scipy `idct` while its hand-written backward applies the unnormalized `dct`, which is not its adjoint; the option is disabled in every script and absent from the paper, so it is not provided.
- The training criterion is the configured loss (the official criterion is MSE).

## Verification

Checked properties: the DCT matrix against scipy's orthonormal DCT-II and its orthogonality (Eq. 7); the `2/L` scaling of Eq. 4; the Haar approximation with odd-length zero padding; an end-to-end comparison with a numpy/scipy re-derivation of the official forward; the low-rank factor shapes and rank of Eq. 5; the ablation switches; level-shift equivariance from the mean centering; the training-only L1 forecast penalty; gradients to both factors.

## Citation

```bibtex
@article{dey2025hadl,
  title   = {HADL Framework for Noise Resilient Long-Term Time Series Forecasting},
  author  = {Dey, Aditya and Kusch, Jonas and Al Machot, Fadi},
  journal = {arXiv preprint arXiv:2502.10569},
  year    = {2025}
}
```
