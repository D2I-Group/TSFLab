# TSRM — reference

## Paper

TSRM: A Lightweight Temporal Feature Encoding Architecture for Time Series Forecasting and Imputation (arXiv 2504.18878, 2025). Sources used: Sec. 2.1, Eqs. 1-3, Sec. 3.1-3.2, Appendix E.

```bibtex
@article{leppich2025tsrm,
  title   = {TSRM: A Lightweight Temporal Feature Encoding Architecture for Time Series Forecasting and Imputation},
  author  = {Leppich, Robert and Stenger, Michael and Grillmeyer, Daniel and Borst, Vanessa and Kounev, Samuel},
  journal = {arXiv preprint arXiv:2504.18878},
  year    = {2025}
}
```

## Implementation mapping

Independent rewrite after reading the pinned official code (`RobertLeppich/TSRM` at `df90daba`, Apache-2.0): `architecture/model.py` (`EncodingLayer`, `PositionalFeedforward`, `Transformations`, `TSRM`, `TSRMForecasting`), `architecture/multiHeadAttention.py`, `architecture/RevIN.py`, `architecture/loss_functions.py`, `embedding/data_embedding.py`, `experiments/setups.py`, and `experiments/configs/forecasting_*.yml`. Nothing was copied or imported; 1.5-entmax is written from its definition (Peters et al., 2019), not from the vendored `architecture/entmax.py`.

Resolved from the official code:

- Affine RevIN on the raw window, then a shared bias-free `Linear(1 -> d)` per value; no positional embedding.
- RL convolutions keep `d` channels (`n_kernel = 1`), stride = kernel size, no padding, groups `-1` = depthwise, ELU afterwards.
- The first layer stores its representation as the global residual; later layers add the running residual to their representation, and every layer adds its post-block representation to it.
- Both blocks are pre-activation (one-group `GroupNorm`, GELU) with dropout on the sublayer output.
- Attention: biased Q/K/V/output projections, `1/sqrt(d_k)` scaling, softmax or 1.5-entmax weights, dropout on the weights. The official option `classic` is named `softmax` here.
- Feed-forward: `ReLU(W_2 W_1 x)`.
- ML: transposed convolutions with matching kernel, stride, dilation, and groups; output padding restores `T` when `T - delta (s - 1) - 1` is not a multiple of the stride; `Linear(K d -> d)` fusion.
- Head: one shared `Linear(T d -> H)` without dropout. TSRM_IFC (`feature_ff`) applies the feed-forward to the `F x d` features of each position.
- Defaults: the single-valued ETTh1 setup (`h = 16`, `N = 7`, `d = 64`, `entmax15`, kernels 3/5/10 with dilations 1/2/3, depthwise, dropout 0.25); other datasets list search grids instead of a selected configuration.

## Differences in detail

- Embedding: the paper lifts each value, adds a positional embedding, then applies RevIN; the code (followed here) applies RevIN to raw values first and adds no positional embedding. The official `DataEmbedding` builds one `Linear(1, d)` per feature but only ever uses the first, and its optional time embedding cannot broadcast; TSFLab uses one shared lift and no time embedding.
- Normalization: the paper says layer normalization; the code uses `nn.GroupNorm(num_groups=1)` (followed).
- Second block: the paper names one linear layer; the code uses two with a ReLU after the second (followed).
- The paper's switch to deactivate ML gradients does not exist at this revision; the ML is always trained.
- Pooled representations (`add_pooled_representations`) never reach the layer output in the official code; not implemented.
- The forecasting configs hold fractional kernel sizes that `nn.Conv1d` cannot use; `experiments/setups.py` overrides them with integer kernels, which TSFLab follows. The shipped kernels `(10, dilation 3)` or `(15, dilation 3)` cover 28 or 43 of 96 steps, below the 50-80% stated in Appendix E.
- The official runner drops the last incomplete test batch and reshapes by the configured batch size; TSFLab takes shapes from the input and evaluates with the catalog runner.
- Also not implemented: `propsparse` attention, the time-feature embedding, imputation and pretraining heads, `n_kernel > 1`.
