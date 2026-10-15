# AdaMamba — reference

## Paper

AdaMamba: Adaptive Frequency-Gated Mamba for Long-Term Time Series Forecasting (arXiv 2604.23239, 2026-04; Sections 3.3-3.5, Eqs. 2-20, Algorithm 1).

```bibtex
@article{jiang2026adamamba,
  title   = {AdaMamba: Adaptive Frequency-Gated Mamba for Long-Term Time Series Forecasting},
  author  = {Jiang, Xudong and Loo, Mingshan and Yang, Hanchen and Li, Wengen and Zhang, Mingrui and Zhang, Yichao and Guan, Jihong and Zhou, Shuigeng},
  journal = {arXiv preprint arXiv:2604.23239},
  year    = {2026}
}
```

## Implementation mapping

- `InteractionEncoding` is Eqs. (2)-(3); `MultiScalePatchEmbedding` is Eqs. (4)-(5) with `(L - P) // s + 2` patches per scale and no positional embedding.
- `FrequencyGatedSSM.frequencies` is Eqs. (6)-(8) (two-layer ReLU adapter on the token mean); `FrequencyGatedSSM.forward` is Algorithm 1 (forgetting gate Eq. 17, cos/sin modulation Eqs. 10-11, amplitude Eq. 12, output Eqs. 14-16).
- `Model.forward` stacks layers with residual and dropout, then `flatten_forecast_head` (Eq. 20) inside affine `revin`.
- Official files read (nothing copied or imported): `models/AdaMamba.py` (model, encoder, `SFMMBlock`, parallel scan), `layers/Embed.py` (`MultiScalePatchEmbedding`, `PatchEmbeddingNG`), `layers/RevIN.py`, `run_longExp.py`, `scripts/AdaMamba_ETTh1.sh`.

## Differences in detail

- Resolved from the official code: the interaction weight multiplies the input (`beta`, initialised to 1, so training starts from the identity mixture); right replicate padding by the stride and no positional embedding; time inside the scan is `m / M` over the concatenated multi-scale sequence; `omega_base` is learnable and `omega` is clamped at 0; the state grid has one row per hidden unit `d` and one column per frequency `S`; the input drive is `sigmoid(gate) * tanh(candidate)` per `(d, S)` cell; the output gate is per frequency and the amplitude projection is `tanh(W_s E_s + b_s)`; every layer adds a residual and dropout; the head is a shared flatten-linear head followed by `head_dropout`; RevIN is affine.
- Initialisation: Linear weights Xavier-uniform with gain 0.1 and zero biases; the raw output parameters (`U_o`, `V_o`, `W_z`) keep their standard-normal initialisation, reproducing the effective behaviour of the name-based official `reset_parameters`.
- Defaults follow the shipped script and parser: `d_model = S = 32`, `e_layers = 2`, patch lengths 96/18/9, stride 16, Conv1D kernel 3, dropout 0.05, head dropout 0.
- Paper vs this entry (code followed): `z_{m-1}` in the gates of Eqs. (15), (18), (19) is replaced by the previous input token (zero at the first step); the input enters through a learned gated drive per `(d, S)` cell instead of `B_m u_m`; no `D^u u_m` or `D^y y_{m-1}` terms in Eq. (14); the trigonometric argument uses `m / M` rather than `m`; the phase (Eq. 13) is not used, as the paper also states.
- The official parallel scan is replaced by an exact sequential scan (`linear_scan`) with the same values and gradients. Unused official modules (phase projection, `freq_weights`, `freq_attn`, `proj`, `alpha`, `gama`) are omitted.
- Training uses the catalog trainer and configured loss (the official runs use MSE, Adam, and `type1` learning-rate decay).

## Verification

The interaction blend, multi-scale patch count and replicate padding, the sequential scan against the closed-form recurrence, the adaptive frequency bases and clamp, the forgetting gate and cos/sin state update against a step-by-step reference of Algorithm 1, the output gate and frequency sum, and the official initialisation were checked. Reported benchmark numbers are not reproduction claims of this implementation.
