# LatentTSF — reference

## Paper

Yang et al., ICML 2026 (arXiv 2602.00297, https://arxiv.org/abs/2602.00297).

Accurate forecasters often learn temporally disordered latent representations ("Latent Chaos"), attributed to
point-wise regression on noisy, partially observed data. LatentTSF shifts forecasting from observation regression
to latent state prediction: an autoencoder projects each observation into a latent state space and forecasting
happens entirely there. An information-theoretic analysis motivates the latent objectives as surrogates for mutual
information between predicted and true latent states and future observations; benchmarks show gains in accuracy and
representation quality.

## Two-stage implementation

- Stage 1 (pretrain): a per-timestep MLP autoencoder `(E, D)` maps each `enc_in`-dim observation to a
  `d_model`-dim latent, is pretrained by reconstruction (`ae_loss = "MAE"` default, `ae_lr`), then frozen. It runs
  once per run through the `pretrain(train_loader, device)` hook called by `run_one`.
- Stage 2 (forecast): DLinear (the paper's primary Table-1 backbone) maps `X -E-> Z_X -f-> Ẑ_Y -D-> Ŷ` with
  `Z_Y = E(Y)`. Latent terms of Eq. 5: `mse_weight·‖Z_Y-Ẑ_Y‖² + cosine_weight·(1-cos)`, defaults `mse_weight = 10`
  (α) and `cosine_weight = 15` (β). The perceptual loss is off, matching Sec. 5.3.1; early stopping uses
  observation-space MSE.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2602-00297,
  author  = {Jie Yang and Yifan Hu and Yuante Li and Kexin Zhang and Kaize Ding and Philip S. Yu},
  title   = {From Observations to States: Latent Time Series Forecasting},
  journal = {CoRR},
  volume  = {abs/2602.00297},
  year    = {2026},
  doi     = {10.48550/ARXIV.2602.00297}
}
```
