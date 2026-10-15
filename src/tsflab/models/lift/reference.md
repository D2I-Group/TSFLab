# LIFT — reference

## Differences in detail

Inputs are `[B, seq_len, enc_in]`; marks and decoder inputs are ignored; the output is `[B, pred_len, enc_in]`. `kernel_size` must be odd (it is the DLinear moving-average width). The official repository (`SJTU-DMTai/LIFT`, formerly `SJTU-Quant/LIFT`) was read at the pinned revision (`models/LIFT.py`, `util/lead_estimate.py`, `models/DLinear.py`).

- **Backbone.** Fixed to the shared DLinear component (shared weights across channels, `kernel_size` 25 by default) and trained jointly with the refiner. Official LIFT wraps any backbone and recommends a pretrained, frozen backbone (`--pretrain --freeze`); neither a frozen backbone nor other backbones are provided here.
- **Online lead estimation.** Official code precomputes leaders, leading steps and correlations over the whole dataset (`prefetch/`) and caches frozen-backbone predictions; here they are estimated online from each normalized lookback window with the same rules (circular FFT cross-correlation, local maxima only, lag 0 excluded, top-K by absolute correlation, step offset +1, sign flip for negative correlation, a variate may lead itself). As in the official code, correlations are evaluated in chunks of `lead_chunk_size` target variates (default 32), so the full `[B, C, C, L]` tensor is never materialized (peak `[B, chunk, C, L]`); the result is identical for any chunk size.
- **Filter weights.** The constant-one logit that competes with the `|corr|` logits in the temperature softmax follows the official code rather than the paper text; the official README notes the method was slightly revised after submission.
- **State filters.** With `state_num=1` official code uses a state-free linear filter factory; here the state classifier branch always runs (a softmax over one state is constant, so the function is the same with unused parameters).
- **Initialisation.** The complex mixing map uses uniform real and imaginary parts bounded by `1/sqrt(in_features)`; the official one uses a complex `nn.Linear` initialisation (or a kaiming-initialised `ComplexLinear` under distributed training). The state prior, state bias and filter factory use the official bounds.
- **Training.** Loss, optimiser, schedule, early stopping and per-dataset hyperparameters are runner configuration. The preset `leader_num=4` and `state_num=8` match the official README example.

## Citation

```bibtex
@inproceedings{LIFT,
  title     = {Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators},
  author    = {Lifan Zhao and Yanyan Shen},
  booktitle = {The Twelfth International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=JiTVtCUOpS}
}
```
