# APTF — reference

## Paper

- **Title**: Amortized Predictability-aware Training Framework for Time Series Forecasting and Classification
- **Venue**: WWW 2026 (The ACM Web Conference; arXiv 2602.16224, 2026-02)
- **arXiv**: https://arxiv.org/abs/2602.16224

```bibtex
@inproceedings{zhang2026aptf,
  title     = {Amortized Predictability-aware Training Framework for Time Series Forecasting and Classification},
  author    = {Zhang, Xu and Wang, Peng and Li, Yichen and Wang, Wei},
  booktitle = {Proceedings of the ACM Web Conference 2026},
  year      = {2026}
}
```

## Implementation mapping

- `predictability_aware_loss` is Algorithm 1; `bucket_weights` gives 1 decreasing by `1/(K-1)`, the last half of the previous one; buckets hold `floor(N/K)` samples with the remainder in the last.
- `hierarchical_loss` is Sec. 3.1.4: group `i` uses `K-i` buckets and the weights left after removing the `i` largest.
- `stage` is Eq. (2); `training_setup` records the batches per epoch and the objective counts `train_step` (saved with checkpoints).
- `training_objective` is Algorithm 2 with the `amortizer` model.
- Official files read (nothing copied): `Amortized_Hierarchical_Predictability_Aware_Loss.py`, `exp/exp_main_public.py`, `Main_LongTerm_TSF.py` (`Meteor-Stars/APTF`, revision `049bc9b3`).

## Differences in detail

- Resolved from the official code: samples are ranked by the per-sample mean squared error over horizon and channels; buckets are filled `floor(N/K)` at a time with the remainder in the last and empty buckets skipped; each bucket's loss is the mean error of its samples times its weight, summed over buckets; the accumulated groups are divided by the fixed maximum number of groups (not the current stage); the source and amortization models bucket each other's errors on the same batch and each is optimized only on its own term; inference and validation use the source model.
- Backbone defaults follow the official long-term script's PatchTST setting for ETTm1 (`d_model=128`, `n_heads=16`, `d_ff=256`, dropout 0.2, three encoder layers, patch 16, stride 8).
- Paper over code: the first stage has `K = 9` buckets (official `K - 1 = 8`), so there are `G = K` stages.
- Bucket losses use the configured criterion (MSE in the paper; set `training.loss = "mse"` to match); ranking always uses per-sample MSE.
- The two models share the catalog optimizer (official: one AdamW per model with its own schedule); with disjoint parameters and separate loss terms the gradients are the same, but adaptive optimizer state and gradient clipping act on the joint parameter set.
- The backbone is the shared `patchtst` component, one of the paper's eleven backbones; its dropouts collapse to one rate.
- The epoch is derived from training steps and loader length, assuming one objective call per training micro-batch.
- The paper's time-series classification variant is not part of this entry. Reported benchmark numbers are not reproduction claims of this implementation.
