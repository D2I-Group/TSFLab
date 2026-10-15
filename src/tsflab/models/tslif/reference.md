# TSLIF — reference

## Paper

TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting (ICLR 2025, arXiv 2503.05108).

```bibtex
@inproceedings{feng2025tslif,
  title     = {TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting},
  author    = {Feng, Shibo and Feng, Wanjin and Gao, Xingyu and Zhao, Peilin and Shen, Zhiqi},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year      = {2025}
}
```

## Implementation mapping

Independent rewrite from Section 4.1 (Eqs. 5-6) and Appendices A.3 and A.12 after reading the pinned official code (`kkking-kk/TS-LIF` at `a59826a6`): `TS-LIF/SeqSNN/network/snn/TSLIF.py`, `spikegru.py`, `surrogate.py`, `SeqSNN/runner/vsts.py`, `SeqSNN/runner/timeseries.py`, and `exp/forecast/spikegru/spikegru_electricity.yml`. The repository root has no license file; the code directory carries the MIT license of Microsoft SeqSNN, recorded as `MIT`. Nothing was copied or imported.

Resolved from the official code:

- Encoder: `Conv2d(1, hidden, (1, 3))`, padding 1, BatchNorm2d, then a single-step LIF, so it fires `H(BN(conv) - 1)`.
- GRU layer: gates recomputed from the current input and a zero hidden state at every step (only the hidden-to-hidden bias contributes), Heaviside gates with an arctangent surrogate (`alpha = 2`); the temporal memory is the TS-LIF membrane state, reset at the start of each window.
- Channels are flattened into the batch; the head maps the last step's `hidden` spikes of each channel to the horizon.
- Inputs are standardized per window (population variance, `eps = 1e-5`, non-affine) and the forecast is denormalized.
- `hidden_size = 128`, one layer, threshold 1.0 (Tables 4-5).
- Neuron initial values: dendrite decay 0.8 with input gain 0.2, soma decay 0.3 with input gain 0.7, couplings -0.1 (soma to dendrite) and -0.8 (dendrite to soma), resets 0.5 and the threshold.

## Differences in detail

- Backbone: at the pinned revision only the GRU path runs with TS-LIF (the configurations tagged `tslif` are `spikegru_*`). The neuron's mixing weights are hard-coded to width 128, which does not broadcast against the TCN `[B, H, C, L]` tensors, and the attention module references undefined neuron classes. Here mixing is per feature, sized from the hidden width.
- Neuron: the official code learns the input gains separately from `1 - alpha` (initialized to the same values), subtracts a fixed 0.5 and `v_th` right after firing, and mixes dendrite and soma spikes with two free per-feature weights drawn from a standard normal. TSFLab ties the gains, subtracts learnable resets `gamma` from the pre-reset potential in the next update, and uses `kappa s_d + (1 - kappa) s_s` with `kappa = 0.5` initially.
- Encoder: plain LIF threshold as in the code, although Appendix A.12 names the TS-LIF neuron.
- GRU cell: the hidden-to-hidden weight matrix receives no signal in the official cell; it is replaced by its bias, keeping the zero-state behaviour.
- Not implemented: the official missing-value injection (ratio 0) and the TCN/Transformer backbones.
