# MAGE — reference

## Paper

Ma et al., NeurIPS 2025. STGNN quality hinges on the learned graph, but adaptive graph learning is costly and
limited in expressiveness; the paper shows ReLU in existing methods amplifies edge-level noise. MAGE uses a sparse,
balanced mixture of experts, each perceiving its own graph through kernel functions with linear complexity in the
number of nodes, and proves one convolution on the learned graph equals several steps on conventional graphs.
It is competitive with strong computational efficiency on several real-world spatiotemporal datasets.

## Differences in detail

Each expert performs node-to-basis-to-node kernel propagation in linear node complexity; top-k routing is combined
with a small balancing path, and three recurrent depths feed the residual forecast.

`MixtureGraphBlock`'s gate was compared against the `topk_expert_router` component: it selects top-k on raw logits,
then applies a masked softmax (sums to one only over selected experts) and blends 5%/95% with the batch/node-mean
dense softmax. That is a different routing formula from the floor-renormalized `topk_dense_mix` used by DUET and
DynamicTMoE, so it stays model-local rather than forcing a flag-driven shared abstraction.

## Citation

```bibtex
@inproceedings{ma2025less,
  author    = {Jiaming Ma and Binwu Wang and Guanjun Wang and Kuo Yang and Zhengyang Zhou and Pengkun Wang and Xu Wang and Yang Wang},
  title     = {Less but More: Linear Adaptive Graph Learning Empowering Spatiotemporal Forecasting},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025},
  url       = {https://github.com/PoorOtterBob/MAGE}
}
```
