# MixLinear — reference

## Paper

Ma, Luo, Sha, ICLR 2026 (arXiv 2410.02081, 2024-10).

Transformers are accurate for long-term forecasting but too compute-intensive for constrained hardware, while linear
models use time-domain decomposition or compact frequency representations. MixLinear models intra- and inter-segment
variation in the time domain and extracts frequency variation from a low-dimensional latent space, reducing a
downsampled n-length one-layer linear model from O(n^2) to O(n) parameters. On four benchmarks it matches or beats
state-of-the-art models with about 0.1K parameters.

## Citation

```bibtex
@misc{ma2024mixlinear,
  author        = {Aitian Ma and Dongsheng Luo and Mo Sha},
  title         = {MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters},
  year          = {2024},
  eprint        = {2410.02081},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2410.02081}
}
```
