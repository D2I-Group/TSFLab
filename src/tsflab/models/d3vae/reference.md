# D3VAE — reference

## Implementation details

- Independent PyTorch implementation; no official code was copied. The official repository (Apache-2.0,
  PaddlePaddle) was checked at the pinned revision in `research/D3VAE/model/model.py`,
  `model/diffusion_process.py`, `model/encoder.py`, `model/neural_operations.py`, `model/resnet.py`,
  `model/embedding.py`, `exp/exp_model.py`, `main.py`, together with the paper's Sec. 2, Eq. 2-14,
  Algorithms 1-2 and Appendix C.
- `Model.representation` builds `X_input = concat(GRU(E(X)), E(X))` (Appendix C.1) with the `embed`
  token/position embeddings plus a model-local learned calendar embedding, treated as a one-channel
  `[time, feature]` grid. The `embed` token convolution is bias-free and Kaiming-initialized (the official one
  has a bias, which the calendar projection bias absorbs when marks are present); raw marks are adapted with `marks`.
- `BidirectionalVAE` is NVAE-style: a bottom-up encoder with a stride-2 preprocessing block, a top-down
  decoder drawing each latent group `z_i ~ p_phi(z_i | z_<i, X)` through encoder/decoder combiners, and a
  projection over the feature axis emitting a Gaussian over the target.
- Training (`Model.training_loss`) diffuses the representation with `beta` and the target with `omega * beta`
  at one random step (Eq. 3-6) and optimizes Eq. 14 = `psi * NLL + lambda * DSM + gamma * TC + MSE`.
- `EnergyNetwork` (residual convolutions, quadratic read-out) is trained by the multi-scale denoising score
  matching of Eq. 10 on detached generated samples, weighted by `sigma_t = 1 - abar_t`.
- `Model.forward` runs Algorithm 2 without diffusion and returns `Y_clean = Y_hat - sigma0^2 grad E(Y_hat)`
  (Eq. 11); `forecast_with_uncertainty` also returns the generated mean and the noise estimate.
- Output `[B, pred_len, D]` with `D = enc_in` for `M`/`S` and `D = 1` (last channel) for `MS`, the official
  target. It is stochastic through latent sampling, so no distribution output is declared.
- Defaults follow the official `main.py`: `res_mbconv` cells, 1 preprocessing block of 3 cells, 2 latent
  groups of 8 channels, 32 channels, embedding 64, GRU 128 x 2, `T = 100`, `beta` in `[0, 0.01]`,
  `omega = 0.1`, `psi = 0.5`, `lambda = 1`, `gamma = 0.01`, `sigma0^2 = 0.001`.

## Differences in detail

- **Horizon length.** The paper and code use `seq_len == pred_len`. When they differ, a model-local
  `Linear(seq_len -> pred_len)` over time maps the representation to the horizon before the BVAE.
- **Noise.** The official port draws diffusion noise with `randint_like(x, 0, 1)` (zeros); Gaussian noise is
  used here as the paper states. As officially, input diffusion acts on the GRU/embedding representation.
- **Score matching.** The energy gradient is built with `create_graph=True` so Eq. 10 trains `zeta` (the
  official `paddle.grad` call does not); DSM noise levels use `sigma_t = 1 - abar_t` (Eq. 10), not the official
  extra `beta * 0.5` schedule.
- **Total correlation.** The paper sign (`+ gamma * L_TC`) is used; the code subtracts it. The official
  estimator (log-sum-exp over latent channels, min-max normalized over the batch, second chunk read as a
  log-variance) is reproduced with an epsilon guard; Eq. 13 averages over all groups (the code skips the first
  group but still divides by the group count).
- **Gaussian sampling.** `mu + exp(log_sigma) * eps` with `tanh`-clamped parameters (`clamp`, bound 1), matching
  the NLL; the official sampler uses `exp(0.5 * exp(log_sigma))`.
- **Network details.** Swish follows every BatchNorm as in NVAE (Paddle `SyncBatchNormSwish` applies none);
  BatchNorm momentum 0.05 in PyTorch convention; weight-norm gains and the decoder's initial state are trainable
  (plain tensors officially). Unused official modules (prior samplers, discriminator, extra GRU/projection) are
  omitted. The energy network keeps the official resolution and flattens per sample, matching the official
  reshape for the 16 x 1 target grid.
- **Calendar embedding.** Learned month/day/weekday/hour (and quarter-hour for `freq = "t"`) tables from raw
  marks; the official loader casts continuous time features to integers instead.
- **Training setup.** Official training uses Adam (lr 0.005 in `main.py`, 5e-4 in the paper), batch 16,
  gradient clipping 1.0 and sliced short datasets; here the repository trainer applies and validation uses the
  configured criterion on `forward`. Published metrics are not claimed.

## Citation

```bibtex
@inproceedings{li2022generative,
  title     = {Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement},
  author    = {Yan Li and Xinjiang Lu and Yaqing Wang and Dejing Dou},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2022},
  url       = {https://arxiv.org/abs/2301.03028}
}
```
