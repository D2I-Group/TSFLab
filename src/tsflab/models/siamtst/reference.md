# SiamTST — reference

## Differences in detail

- Official sources read (revision `f8069d507c22a46bf3546c89a75a9da5ba055a17`, Apache-2.0): `core/model.py`, `core/loss.py`, `core/layers/attention.py`, `core/layers/transformer.py`, `core/layers/ffn.py`, `core/layers/embedding.py`, `core/layers/patch.py`, `core/layers/heads.py`, `core/layers/norm.py`, `core/data/datasets.py`, `core/utils.py`, `pretrain.py`, `finetune.py`, `train.py`; nothing copied or imported.
- Resolved from the official code: patches are taken from the last `P + S (K - 1)` steps; the patch projection keeps its bias and PyTorch default initialization; the positional table is `U(-0.02, 0.02)` (paper says `U(0, 0.2)`).
- RMSNorm has a learnable weight and `eps = 1e-5`; QK-Norm normalizes each head's queries and keys over the head width (per-head value width `d_model / n_heads`, correcting Eq. 4).
- Attention and feed-forward outputs each pass dropout before their residual; the feed-forward block has dropout after GELU and after the second map; a final RMSNorm follows the stack.
- Masking zeroes `K - int(K (1 - ratio))` random patches per series with one ratio per batch; the reconstruction head is `Dropout -> Linear(d_model, patch_size)` per patch; the loss averages squared error over each patch, then over masked patches.
- Pre-training: AdamW (`lr 1e-3`, weight decay 0.1, betas 0.9/0.98), one-cycle schedule, `alpha = 0.2`; fine-tuning freezes every parameter but the forecasting head.
- Hyperparameter defaults follow the code (`d_model 64`, 4 heads, patch 16 stride 16, 4 layers, `d_ff = 4 d_model`), not PatchTST's published settings as Sec. 4.3 claims.
- Shared components: `revin.RevIN(affine=False)` matches `RevInNorm` in `core/layers/norm.py` (detached mean and population std with `eps = 1e-5` inside the square root, inverted on the forecast). RMSNorm is `torch.nn.RMSNorm` (`x / sqrt(mean(x^2) + eps) * weight`).
- The catalog has no separate pre-training dataset, so `pretrain` uses the run's training loader; the official script keeps the best-validation pre-training checkpoint, here the final state is used.
- Fine-tuning uses the catalog trainer (the official stage uses MSE, AdamW, and a one-cycle schedule for 30 epochs). Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@article{kristoffersen2024siamtst,
  title   = {SiamTST: A Novel Representation Learning Framework for Enhanced Multivariate Time Series Forecasting applied to Telco Networks},
  author  = {Kristoffersen, Simen and Nordby, Peter Skaar and Malacarne, Sara and Ruocco, Massimiliano and Ortiz, Pablo},
  journal = {arXiv preprint arXiv:2407.02258},
  year    = {2024}
}
```
