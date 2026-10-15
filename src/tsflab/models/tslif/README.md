---
name: "TSLIF"
description: "Channel-independent spiking GRU whose dual-compartment TS-LIF neurons mix slow dendritic and fast somatic spikes, with a linear head on the last step. Use for spiking-network baselines on per-channel univariate dynamics; not for cross-channel dependence or probabilistic output."
---

# TSLIF

## Idea

- `TSLIFNeuron` (Eqs. 5-6): a dendrite (`alpha_d`) and a soma (`alpha_s`) both take `(1 - alpha) c[t]`, couple through `beta_d v_s[t-1]` and `beta_s v_d[t]`, subtract learnable resets, fire `H(v - v_th)`, and output `kappa s_d + (1 - kappa) s_s`.
- `ATanSpike` is the Heaviside spike with an arctangent surrogate gradient, used for every spike and gate.
- `ConvSpikeEncoder` (Appendix A.12): a `1 x kernel` time convolution shared by channels, BatchNorm, and a threshold spike per hidden map and step.
- `SpikingGRULayer` (SeqSNN spiking GRU): spike-valued gates from the input with a zero hidden state, `h = (1 - z) n`, fed as current to a TS-LIF neuron whose membrane state runs over the window.
- Non-affine `revin`, channel-independent stepping, and a linear map from the last step's top-layer spikes to the horizon.

## When to use

- Each channel is modelled alone with shared weights, so it suits sets of weakly related channels.
- The two compartments decay at different rates, giving slow and fast temporal memory inside one neuron.
- Not for tasks needing cross-channel modelling, covariates, or quantiles (point output only).

## Configure

- `enc_in`: must equal the dataset channel count.

Other hyperparameters: preset defaults in `configs/models/TSLIF.toml`; tune generically.

## Differences

- Only the TS-GRU backbone is implemented; the paper's TCN and Transformer backbones are not runnable with TS-LIF in the official code.
- The neuron follows the paper equations, not the code: input gains tied to `1 - alpha`, learnable resets subtracted in the next update, and a convex `kappa` mix initialized at 0.5.
- The encoder fires with a plain LIF threshold as in the official code (the appendix names TS-LIF there).
- The dead hidden-to-hidden weight of the official GRU cell is replaced by its bias alone.
- Missing-value injection is not implemented; no dependency on `snntorch` or `spikingjelly`. Reported benchmark numbers are not reproduced.
