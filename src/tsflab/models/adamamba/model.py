"""Paper-driven local implementation of AdaMamba.

AdaMamba (Jiang et al., arXiv 2604.23239) forecasts a RevIN-normalized window in
two modules. The interactive patch encoding module (Sec. 3.3) mixes the input
with a channel-mixing ``Conv1D`` through a learned scalar (Eqs. 2-3) and embeds
channel-independent patches of several lengths that are concatenated along the
patch axis (Eqs. 4-5). The adaptive frequency-gated state-space module (Sec. 3.4,
Algorithm 1) runs, per channel, a selective recurrence over the patch sequence
whose state is a ``[d, S]`` time-frequency grid: the frequency bases
``omega = omega_base + Adapter(AvgPool(U_d))`` (Eqs. 6-8) modulate the input by
``cos(omega t)`` and ``sin(omega t)`` (Eqs. 10-11), the forgetting gate is the
outer product of a temporal and a frequency gate (Eqs. 17-19), and the amplitude
of the complex state (Eq. 12) is projected per frequency and summed under an
output gate (Eqs. 14-16). A flatten head maps the representation to the horizon
(Eq. 20). Where the paper and the pinned official code differ, this module
follows the code; the model card lists each difference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.flatten_forecast_head import FlattenForecastHead
from tsflab.models._components.revin import RevIN


def patch_count(seq_len: int, patch_len: int, stride: int) -> int:
    """Patches of one scale after right replicate padding by ``stride``."""
    return (seq_len + stride - patch_len) // stride + 1


def linear_scan(decay: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """``h_m = decay_m * h_{m-1} + inputs_m`` with ``h_{-1} = 0`` along axis 1."""
    state = torch.zeros_like(inputs[:, 0])
    states = []
    for step in range(inputs.shape[1]):
        state = decay[:, step] * state + inputs[:, step]
        states.append(state)
    return torch.stack(states, dim=1)


class InteractionEncoding(nn.Module):
    """Eqs. (2)-(3): ``beta * X + (1 - beta) * Conv1D(X)`` on ``[B, N, L]``.

    The convolution mixes the ``N`` variables over a short time kernel. The
    paper's alpha weights the convolution; the official code learns the weight
    of the input, ``beta = 1 - alpha``, initialised to 1.
    """

    def __init__(self, enc_in: int, kernel_size: int, beta: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(enc_in, enc_in, kernel_size, padding="same")
        self.beta = nn.Parameter(torch.tensor([float(beta)]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.beta * x + (1.0 - self.beta) * self.conv(x)


class MultiScalePatchEmbedding(nn.Module):
    """Eqs. (4)-(5): per scale, right replicate padding by ``stride``, unfold, Linear(P_i -> d); concat patches."""

    def __init__(self, d_model: int, patch_lens: tuple[int, ...], stride: int, dropout: float) -> None:
        super().__init__()
        self.patch_lens = patch_lens
        self.stride = stride
        self.projections = nn.ModuleList(nn.Linear(p, d_model) for p in patch_lens)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, N, L] -> [B * N, M, d]`` with ``M`` the summed patch count."""
        padded = F.pad(x, (0, self.stride), mode="replicate")
        tokens = []
        for patch_len, projection in zip(self.patch_lens, self.projections):
            patches = padded.unfold(-1, patch_len, self.stride)  # [B, N, Q_i, P_i]
            patches = patches.reshape(-1, patches.shape[2], patch_len)
            tokens.append(self.dropout(projection(patches)))
        return torch.cat(tokens, dim=1)


class FrequencyGatedSSM(nn.Module):
    """Algorithm 1: adaptive frequency-gated state-space update over patch tokens.

    Input and output are ``[B', M, d]`` (one sequence per channel). The state is
    a real and an imaginary ``[d, S]`` grid per step.
    """

    def __init__(self, d_model: int, d_freq: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_freq = d_freq
        # Eq. (6): omega_base = [2 pi k / S]_{k=0}^{S-1}; learnable in the official code.
        self.omega_base = nn.Parameter(2 * math.pi * torch.arange(d_freq, dtype=torch.float32) / d_freq)
        # Eq. (7): Adapter = Linear -> ReLU -> Linear on the mean-pooled tokens.
        self.adapter = nn.Sequential(
            nn.Linear(d_model, d_model // 4), nn.ReLU(), nn.Linear(d_model // 4, d_freq)
        )
        # Eqs. (18)-(19): temporal (per d) and frequency (per S) forgetting gates.
        self.time_gate_z = nn.Linear(d_model, d_model)
        self.time_gate_u = nn.Linear(d_model, d_model)
        self.freq_gate_z = nn.Linear(d_model, d_freq)
        self.freq_gate_u = nn.Linear(d_model, d_freq)
        # Input path of Eqs. (10)-(11): sigmoid gate times tanh candidate per (d, S).
        self.input_gate_z = nn.Linear(d_model, d_model * d_freq)
        self.input_gate_u = nn.Linear(d_model, d_model * d_freq)
        self.candidate_z = nn.Linear(d_model, d_model * d_freq)
        self.candidate_u = nn.Linear(d_model, d_model * d_freq)
        # Output gate (Eq. 15) and per-frequency amplitude projection (Eq. 14).
        self.out_amp = nn.Parameter(torch.randn(d_freq, d_model))
        self.out_u = nn.Parameter(torch.randn(d_freq, d_model))
        self.out_z = nn.Linear(d_model, d_freq)
        self.out_bias = nn.Parameter(torch.zeros(d_freq))
        self.amp_proj = nn.Parameter(torch.randn(d_freq, d_model, d_model))
        self.amp_bias = nn.Parameter(torch.zeros(d_freq, d_model))
        self._init_linear()

    def _init_linear(self) -> None:
        # Official initialisation: Linear weights Xavier-uniform with gain 0.1 and
        # zero biases; the raw output parameters keep their standard-normal draw.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                nn.init.zeros_(module.bias)

    def frequencies(self, x: torch.Tensor) -> torch.Tensor:
        """Eqs. (6)-(8): ``omega = clamp(omega_base + Adapter(mean_m x), min=0)`` -> ``[B', S]``."""
        return (self.omega_base + self.adapter(x.mean(dim=1))).clamp(min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, steps, dim = x.shape
        # The recurrent z_{m-1} of the paper is replaced, as in the official code,
        # by the previous input token (zero at the first step) so every gate is
        # available before the scan.
        previous = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)
        omega = self.frequencies(x)
        times = torch.arange(1, steps + 1, device=x.device, dtype=x.dtype) / steps
        phase = omega[:, None, :] * times[None, :, None]  # [B', M, S]

        time_gate = torch.sigmoid(self.time_gate_z(previous) + self.time_gate_u(x))  # [B', M, d]
        freq_gate = torch.sigmoid(self.freq_gate_z(previous) + self.freq_gate_u(x))  # [B', M, S]
        decay = time_gate[..., :, None] * freq_gate[..., None, :]  # Eq. (17): [B', M, d, S]

        shape = (batch, steps, dim, self.d_freq)
        gate = torch.sigmoid((self.input_gate_z(previous) + self.input_gate_u(x)).view(shape))
        candidate = torch.tanh((self.candidate_z(previous) + self.candidate_u(x)).view(shape))
        drive = gate * candidate
        # Eqs. (10)-(11): real and imaginary states driven by cos / sin modulation.
        real = linear_scan(decay, drive * torch.cos(phase)[:, :, None, :])
        imag = linear_scan(decay, drive * torch.sin(phase)[:, :, None, :])
        amplitude = torch.sqrt(real.square() + imag.square() + 1e-8)  # Eq. (12): [B', M, d, S]

        out_gate = torch.sigmoid(
            torch.einsum("bmds,sd->bms", amplitude, self.out_amp)
            + torch.einsum("bmd,sd->bms", x, self.out_u)
            + self.out_z(previous)
            + self.out_bias
        )  # [B', M, S]
        projected = torch.tanh(torch.einsum("bmds,spd->bmsp", amplitude, self.amp_proj) + self.amp_bias)
        return (out_gate[..., None] * projected).sum(dim=2)  # Eq. (16): sum over S -> [B', M, d]


class Model(nn.Module):
    """AdaMamba forecaster with the four-input TSFLab interface."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 32,
        d_freq: int = 32,
        e_layers: int = 2,
        patch_lens: tuple[int, ...] = (96, 18, 9),
        stride: int = 16,
        conv_kernel: int = 3,
        beta: float = 1.0,
        dropout: float = 0.05,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        patch_lens = tuple(int(p) for p in patch_lens)
        if any(p > seq_len for p in patch_lens):
            raise ValueError("every patch length must be at most seq_len")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.revin = RevIN(enc_in, affine=True)
        self.interaction = InteractionEncoding(enc_in, conv_kernel, beta)
        self.patch_embedding = MultiScalePatchEmbedding(d_model, patch_lens, stride, dropout)
        self.num_patches = sum(patch_count(seq_len, p, stride) for p in patch_lens)
        self.layers = nn.ModuleList(FrequencyGatedSSM(d_model, d_freq) for _ in range(e_layers))
        self.dropout = nn.Dropout(dropout)
        self.head = FlattenForecastHead(False, enc_in, d_model * self.num_patches, pred_len, head_dropout)

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """Stacked frequency-gated layers with a residual and dropout after each."""
        for layer in self.layers:
            tokens = self.dropout(layer(tokens) + tokens)
        return tokens

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        x = self.revin(x_enc, "norm").transpose(1, 2)  # [B, N, L]
        batch, channels, _ = x.shape
        tokens = self.patch_embedding(self.interaction(x))  # [B * N, M, d]
        z = self.encode(tokens).reshape(batch, channels, self.num_patches, -1)
        forecast = self.head(z.transpose(2, 3)).transpose(1, 2)  # Eq. (20): [B, H, N]
        return self.revin(forecast, "denorm")
