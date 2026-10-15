# STOP — reference

## Paper

Robust Spatio-Temporal Centralized Interaction for OOD Learning (Ma et al., ICML 2025, PMLR v267).

Node-to-node message passing in spatio-temporal GCNs is sensitive to spatio-temporal shifts. STOP
uses a centralized messaging mechanism (Context-Aware Units) that blocks node-to-node messages, a
message perturbation mechanism that generates variant environments, and a spatio-temporal
distributionally robust optimization over them. Against 14 baselines on six datasets it reports up
to 17.01% better generalization and 18.44% better inductive-learning performance.

## Implementation notes

- All spatial interaction goes through the ConAU bank (`core` units, `head` attention heads); no adjacency is used.
- Calendar prompts are averaged over the lookback and broadcast to every node.
- `environment_forecasts(x, marks, environments=3)` masks the top-scoring nodes' messages per environment and returns `[B, E, pred_len, N]` for an external worst-loss step.

## Differences in detail

The official training step (`LargeST/src/engines/stop_engine.py`, same in `TrafficStream/`)
optimizes the plain masked loss on `pred_corr`, with the worst-environment variance terms commented
out, so the paper's weighting cannot be recovered from code. TSFLab therefore exposes
`environment_forecasts` and leaves the DRO selection to an external loop. The official repository
has no license file (recorded as `NOASSERTION`); nothing was copied.

## Citation

```bibtex
@inproceedings{DBLP:conf/icml/MaW0Z0025,
  author    = {Jiaming Ma and Binwu Wang and Pengkun Wang and Zhengyang Zhou and Xu Wang and Yang Wang},
  title     = {Robust Spatio-Temporal Centralized Interaction for {OOD} Learning},
  booktitle = {Forty-second International Conference on Machine Learning, {ICML} 2025},
  series    = {Proceedings of Machine Learning Research},
  year      = {2025},
  url       = {https://proceedings.mlr.press/v267/ma25s.html}
}
```
