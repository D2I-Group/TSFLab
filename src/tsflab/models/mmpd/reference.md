# MMPD — reference

## Paper

- **Title**: MMPD: Diverse Time Series Forecasting via Multi-Mode Patch Diffusion Loss
- **Venue**: ICLR 2026 (OpenReview NEUgHT8dvH)
- **Abstract (shortened)**: MSE assumes a one-mode Gaussian future and struggles when several diverse outcomes are possible. The Multi-Mode Patch Diffusion (MMPD) loss applies to any patch-based backbone that outputs latent future tokens: a diffusion model conditioned on those tokens models the future distribution, with a lightweight Patch Consistent MLP denoiser keeping denoised patches consistent. Multi-mode predictions with probabilities come from an inference algorithm that fits an evolving variational Gaussian Mixture Model during diffusion. On eight datasets it excels at diverse forecasting and matches MSE and Student-T losses on deterministic and probabilistic metrics.

## Differences in detail

**Local implementation: confirmed.** The module was written for TSFLab; no
external source file is copied. `models/loss_funcs/mmpd/mmpd_loss.py` and
`models/loss_funcs/mmpd/gaussian_diffusion.py` were inspected at the pinned
revision to confirm details. The code follows diffusion Eq. (3), the
token/step/left/right Patch Consistent MLP of Eq. (7), AdaLN-MLP Eqs.
(12)-(13), and the deterministic anchor term of Eq. (8).

`diffusion_loss` is the joint training objective, wired as the runner's
training objective (it replaces the configured criterion during training
only); `sample` exposes conditional reverse trajectories; ordinary `forward`
returns the efficient anchor point forecast required by TSFLab. The evolving
variational-GMM mode fitting from Algorithm 1 and per-mode probabilities are
not part of the common point-forecast output and are not claimed. The local
patch backbone is compact and not a reproduction of every backbone in the
paper.

## Citation

```bibtex
@inproceedings{zhang2026mmpd,
  author    = {Yunhao Zhang and Wenyao Hu and Jiale Zheng and Lujia Pan and Junchi Yan},
  title     = {{MMPD}: Diverse Time Series Forecasting via Multi-Mode Patch Diffusion Loss},
  booktitle = {The Fourteenth International Conference on Learning Representations},
  year      = {2026},
  url       = {https://openreview.net/forum?id=NEUgHT8dvH},
  code      = {https://github.com/Thinklab-SJTU/MMPD}
}
```
