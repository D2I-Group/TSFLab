"""TS-LIF: temporal segment (dendrite + soma) spiking neurons in a spiking GRU.

Independent implementation from Section 4.1 (Eqs. 5-6), Appendix A.3 and A.12 of
Feng, Feng, Gao, Zhao and Shen, "TS-LIF: A Temporal Segment Spiking Neuron Network
for Time Series Forecasting" (arXiv 2503.05108, ICLR 2025), after reading the
pinned official code (``kkking-kk/TS-LIF`` at ``a59826a6``, MIT license file in the
code directory inherited from Microsoft SeqSNN) to resolve omissions; nothing is
copied.

The forecaster is the TS-GRU variant: a convolutional spike encoder per channel,
spiking GRU layers whose output neurons are TS-LIF neurons stepped over the input
window, and a linear head from the last step's spikes to the horizon, wrapped in
non-affine instance normalization.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN


class ATanSpike(torch.autograd.Function):
    """Heaviside spike ``H(x)`` (1 for ``x >= 0``) with the arctangent surrogate
    gradient ``(alpha / 2) / (1 + (pi / 2 * alpha * x) ** 2)``."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (x,) = ctx.saved_tensors
        alpha = ctx.alpha
        surrogate = (alpha / 2) / (1 + (math.pi / 2 * alpha * x) ** 2)
        return grad_output * surrogate, None


def spike(x: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    return ATanSpike.apply(x, alpha)


class TSLIFNeuron(nn.Module):
    """Two-compartment TS-LIF neuron (Eqs. 5-6), all coefficients learnable.

    ``v_d[t] = a_d v_d[t-1] + b_d v_s[t-1] + (1 - a_d) c[t] - g_d s_d[t-1]``
    ``v_s[t] = a_s v_s[t-1] + b_s v_d[t] + (1 - a_s) c[t] - g_s s_s[t-1]``
    ``s_d = H(v_d - v_th)``, ``s_s = H(v_s - v_th)``,
    ``s_mix = kappa s_d + (1 - kappa) s_s`` with one ``kappa`` per feature.
    Initial values follow the official neuron (dendrite decay 0.8, soma decay 0.3,
    couplings -0.1 / -0.8, resets 0.5 / ``v_th``).
    """

    def __init__(self, features: int, v_threshold: float = 1.0, surrogate_alpha: float = 2.0) -> None:
        super().__init__()
        self.v_threshold = v_threshold
        self.surrogate_alpha = surrogate_alpha
        self.alpha_d = nn.Parameter(torch.tensor(0.8))
        self.alpha_s = nn.Parameter(torch.tensor(0.3))
        self.beta_d = nn.Parameter(torch.tensor(-0.1))
        self.beta_s = nn.Parameter(torch.tensor(-0.8))
        self.gamma_d = nn.Parameter(torch.tensor(0.5))
        self.gamma_s = nn.Parameter(torch.tensor(float(v_threshold)))
        self.kappa = nn.Parameter(torch.full((features,), 0.5))

    @staticmethod
    def initial_state(current: torch.Tensor) -> tuple[torch.Tensor, ...]:
        zeros = torch.zeros_like(current)
        return zeros, zeros, zeros, zeros

    def step(
        self, current: torch.Tensor, state: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        """One time step on ``current [..., features]``; ``state`` is
        ``(v_d, v_s, s_d, s_s)`` from the previous step."""
        v_d, v_s, s_d, s_s = state
        v_d = self.alpha_d * v_d + self.beta_d * v_s + (1 - self.alpha_d) * current - self.gamma_d * s_d
        v_s = self.alpha_s * v_s + self.beta_s * v_d + (1 - self.alpha_s) * current - self.gamma_s * s_s
        s_d = spike(v_d - self.v_threshold, self.surrogate_alpha)
        s_s = spike(v_s - self.v_threshold, self.surrogate_alpha)
        mixed = self.kappa * s_d + (1 - self.kappa) * s_s
        return mixed, (v_d, v_s, s_d, s_s)


class ConvSpikeEncoder(nn.Module):
    """Spike encoder (Appendix A.12, Eq. 23): a ``1 x kernel`` convolution over time
    shared by all channels, BatchNorm, and a single-step LIF firing
    ``H(BN(Conv(x)) - v_th)``: ``[B, L, C] -> [B, hidden, C, L]``."""

    def __init__(self, hidden: int, kernel_size: int, v_threshold: float, surrogate_alpha: float) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, hidden, (1, kernel_size), padding=(0, kernel_size // 2))
        self.norm = nn.BatchNorm2d(hidden)
        self.v_threshold = v_threshold
        self.surrogate_alpha = surrogate_alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current = self.norm(self.conv(x.transpose(1, 2).unsqueeze(1)))
        return spike(current - self.v_threshold, self.surrogate_alpha)


class SpikingGRULayer(nn.Module):
    """Spiking GRU layer of the SeqSNN TS-GRU: spike gates from the input and a
    zero hidden state (the hidden-to-hidden map contributes only its bias),
    ``h = (1 - z) n``, then a TS-LIF neuron carries the temporal state."""

    def __init__(self, input_size: int, hidden: int, v_threshold: float, surrogate_alpha: float) -> None:
        super().__init__()
        self.hidden = hidden
        self.surrogate_alpha = surrogate_alpha
        self.input_gates = nn.Linear(input_size, 3 * hidden)
        bound = 1 / math.sqrt(hidden)
        self.hidden_bias = nn.Parameter(torch.empty(3 * hidden).uniform_(-bound, bound))
        self.neuron = TSLIFNeuron(hidden, v_threshold, surrogate_alpha)

    def current(self, x: torch.Tensor) -> torch.Tensor:
        i_r, i_z, i_n = self.input_gates(x).split(self.hidden, dim=-1)
        b_r, b_z, b_n = self.hidden_bias.split(self.hidden)
        r = spike(i_r + b_r, self.surrogate_alpha)
        z = spike(i_z + b_z, self.surrogate_alpha)
        n = spike(i_n + r * b_n, self.surrogate_alpha)
        return (1 - z) * n

    def step(self, x: torch.Tensor, state):
        current = self.current(x)
        if state is None:
            state = self.neuron.initial_state(current)
        return self.neuron.step(current, state)


class Model(nn.Module):
    """TS-LIF (TS-GRU) forecaster ``[B, seq_len, enc_in] -> [B, pred_len, enc_in]``;
    channels share weights and are processed independently after the encoder."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hidden_size: int = 128,
        layers: int = 1,
        kernel_size: int = 3,
        v_threshold: float = 1.0,
        surrogate_alpha: float = 2.0,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = RevIN(enc_in, affine=False)
        self.encoder = ConvSpikeEncoder(hidden_size, kernel_size, v_threshold, surrogate_alpha)
        self.layers = nn.ModuleList(
            SpikingGRULayer(hidden_size, hidden_size, v_threshold, surrogate_alpha) for _ in range(layers)
        )
        self.head = nn.Linear(hidden_size, pred_len)

    def spiking_states(self, x: torch.Tensor) -> torch.Tensor:
        """Last-step output spikes ``[B, C, hidden]`` of the top layer for normalized ``x``."""
        b, length, c = x.shape
        encoded = self.encoder(x)  # [B, H, C, L]
        sequence = encoded.permute(0, 2, 3, 1).reshape(b * c, length, -1)
        states = [None] * len(self.layers)
        out = sequence[:, 0]
        for t in range(length):
            out = sequence[:, t]
            for i, layer in enumerate(self.layers):
                out, states[i] = layer.step(out, states[i])
        return out.reshape(b, c, -1)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.ndim != 3 or x_enc.size(1) != self.seq_len or x_enc.size(2) != self.enc_in:
            raise ValueError(f"TSLIF expects [B, {self.seq_len}, {self.enc_in}]")
        x = self.revin(x_enc, "norm")
        forecast = self.head(self.spiking_states(x))  # [B, C, pred_len]
        return self.revin(forecast.transpose(1, 2), "denorm")
