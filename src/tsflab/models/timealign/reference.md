# TimeAlign — reference

## Paper

Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting (Hu et al., ICLR
2026; arXiv:2509.14181).

TimeAlign explicitly aligns past and future representations to bridge the distribution gap between
input histories and future targets. A lightweight plug-and-play framework, distinct from contrastive
learning, it aligns auxiliary features learned by a simple reconstruction task and feeds them back to
any base forecaster. Eight benchmarks show superior performance, mainly from correcting frequency
mismatches between histories and futures; the paper also justifies theoretically how reconstruction
improves generalization and alignment increases mutual information with the targets.

## Implementation notes

- Local term: `relu(1 - cos(h, f) - local_margin)` averaged over tokens; global term: `relu(|R_h - R_f| - global_margin)` over the token relation matrices; `loc` / `glo` toggle each.
- Future states are detached, so alignment only moves the history branch.
- `pos` adds a learnable position embedding to patch tokens; `layer_norm` toggles LayerNorm in the residual MLP layers.
- The objective needs the future `Y`, which is why it is registered through `ModelSpec.training_objective`.

## Citation

```bibtex
@article{DBLP:journals/corr/abs-2509-14181,
  author  = {Yifan Hu and Jie Yang and Tian Zhou and Peiyuan Liu and Yujin Tang and Rong Jin and Liang Sun},
  title   = {Bridging Past and Future: Distribution-Aware Alignment for Time Series Forecasting},
  journal = {CoRR},
  volume  = {abs/2509.14181},
  year    = {2025},
  doi     = {10.48550/ARXIV.2509.14181}
}
```
