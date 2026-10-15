# TFPS — reference

## Paper

Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift
(Sun et al., NeurIPS 2025; arXiv:2410.09836, revised 2025-10).

## Implementation mapping

- `PatternIdentifier`: `affinity` is Eq. (6), `refined_affinity` Eq. (7), `clustering_loss` Eq. (9) (`R + beta KL(S_hat || S)`, with `penalty` implementing Eqs. 3-5).
- `PatternExperts`: two-layer ReLU MLP experts per patch, aggregated by the top-k softmax gate (Eqs. 10-11).

## Differences in detail

- Official code read at `syrGitHub/TFPS` revision `83a11827e27e6617e8c8a8771f0a1dd7e10976a5`: `models/PatchTST_MoE_cluster.py`, `layers/PatchTST_MoE_backbone_cluster_time.py`, `layers/PatchTST_MoE_backbone_cluster_frequency.py`, `layers/Cluster.py`, `layers/Constraint.py`, `layers/SparseMoE.py`, `exp/exp_main.py`, `run_longExp.py`, `scripts/etth1.sh`.
- Resolved from the official code: RevIN is non-affine and subtracts the mean; each branch has its own patch embedding, learnable uniform(+-0.02) position table and embedding dropout; the time encoder is the PatchTST encoder (post-norm BatchNorm, GELU feed-forward, residual attention scores, dropout on the attention output and sublayers, no attention-weight dropout); the frequency encoder is FNet-style with LayerNorm and an FFT over the hidden axis then the patch axis.
- A linear map over the patch axis takes `N_in` input patches to `N_out = floor((H - P) / S) + 2` output patches before clustering (not described in the paper).
- Every expert is `Linear(q, 4q) -> ReLU -> Linear(4q, q) -> Dropout(0.1)` on the cross-variate patch feature (`q = C D`); the frequency branch's inverse transform is an inverse FFT over the hidden then the patch axis, real part; heads are per variate (`individual = 1`) on the concatenation of both branches.
- Clustering-loss scales: `R = 1e-3 (||D^T D * I - I||_F + ||D^T D * O||_F)` (unsquared norms), KL averaged over its entries with `beta = 0.1`, target `S_hat` computed without gradient, smoothness `eta = 5`, branch losses weighted `alpha = gama = 1` (`lambda_time`, `lambda_freq`). The paper's Eq. (6) sets `eta = d` and Eqs. (3)-(8) use squared, halved norms and a summed KL.
- The official clustering input reshapes `[B, C, D, N_out]` directly to `[B * N_out, C * D]` without a permute; here each row is one patch's cross-variate feature, as in Section 3.5 and the official MoPE input.
- The official `fc_dropout`, decomposition, pretrain/patch-wise heads, and the unused autoencoder are not implemented; learning-rate schedules are run settings.
- Defaults (`e_layers = 3`, `n_heads = 4`, `d_model = 16`, `d_ff = 128`, `dropout = 0.3`, `patch_len = 16`, `stride = 8`, one routed expert) follow the ETTh1 script, which grid-searches 4-16 experts per horizon.
- Reported benchmark numbers are not reproduction claims of this implementation.

## Citation

```bibtex
@inproceedings{sun2025tfps,
  title     = {Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift},
  author    = {Sun, Yanru and Xie, Zongxia and Eldele, Emadeldeen and Chen, Dongyue and Hu, Qinghua and Wu, Min},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2025}
}
```
