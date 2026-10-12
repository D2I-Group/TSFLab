"""Clean-room diagonal S4 forecaster from the continuous-time SSM equations."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._inputs import check_window


def zoh_discretize_diagonal(a: torch.Tensor, b: torch.Tensor, dt: torch.Tensor):
    """Zero-order hold: A_bar=exp(dt A), B_bar=(exp(dt A)-1) A^-1 B."""
    dt_a = dt * a
    a_bar = torch.exp(dt_a)
    ratio = torch.expm1(dt_a) / a
    return a_bar, ratio * b


class DiagonalSSMKernel(nn.Module):
    """Generate K_l = 2 Re(C A_bar^l B_bar) for a conjugate diagonal SSM."""

    def __init__(self, d_model: int, d_state: int, dt_min=1e-3, dt_max=1e-1):
        super().__init__()
        modes = d_state // 2
        initial_dt = torch.empty(d_model).uniform_(math.log(dt_min), math.log(dt_max))
        self.log_dt = nn.Parameter(initial_dt)
        self.log_decay = nn.Parameter(torch.full((d_model, modes), math.log(0.5)))
        frequencies = math.pi * torch.arange(modes).expand(d_model, modes).clone()
        self.frequency = nn.Parameter(frequencies)
        self.b = nn.Parameter(
            torch.view_as_real(torch.ones(d_model, modes, dtype=torch.cfloat))
        )
        c = torch.randn(d_model, modes, dtype=torch.cfloat) / math.sqrt(max(modes, 1))
        self.c = nn.Parameter(torch.view_as_real(c))

    def continuous_parameters(self):
        a = -self.log_decay.exp() + 1j * self.frequency
        return (
            a,
            torch.view_as_complex(self.b.contiguous()),
            torch.view_as_complex(self.c.contiguous()),
        )

    def forward(self, length: int) -> torch.Tensor:
        a, b, c = self.continuous_parameters()
        dt = self.log_dt.exp().unsqueeze(-1)
        a_bar, b_bar = zoh_discretize_diagonal(a, b, dt)
        powers = torch.arange(length, device=a.device, dtype=a.real.dtype)
        impulse = a_bar.unsqueeze(-1).pow(powers)
        return 2.0 * torch.einsum("hn,hn,hnl->hl", c, b_bar, impulse).real


class DiagonalS4Layer(nn.Module):
    def __init__(self, d_model, d_state, dropout):
        super().__init__()
        self.kernel = DiagonalSSMKernel(d_model, d_state)
        self.skip = nn.Parameter(torch.ones(d_model))
        self.output_projection = nn.Conv1d(d_model, 2 * d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values):
        length = values.shape[-1]
        kernel = self.kernel(length)
        spectrum = torch.fft.rfft(values, n=2 * length)
        kernel_spectrum = torch.fft.rfft(kernel, n=2 * length)
        output = torch.fft.irfft(spectrum * kernel_spectrum, n=2 * length)[..., :length]
        output = output + self.skip.view(1, -1, 1) * values
        return F.glu(self.output_projection(self.dropout(F.gelu(output))), dim=1)


class S4ResidualBlock(nn.Module):
    def __init__(self, d_model, d_state, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = DiagonalS4Layer(d_model, d_state, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden):
        update = self.ssm(self.norm(hidden).transpose(1, 2)).transpose(1, 2)
        return hidden + self.dropout(update)


class Model(nn.Module):
    """Forecasting model built from a diagonal approximation to S4."""

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        c_out=None,
        d_model=128,
        d_state=64,
        e_layers=2,
        dropout=0.1,
        use_norm=True,
    ):
        super().__init__()
        c_out = enc_in if c_out is None else c_out
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.use_norm = use_norm
        self.input_projection = nn.Linear(enc_in, d_model)
        self.blocks = nn.ModuleList(
            [S4ResidualBlock(d_model, d_state, dropout) for _ in range(e_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.horizon_projection = nn.Linear(seq_len, pred_len)
        self.output_projection = nn.Linear(d_model, c_out)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        if self.use_norm:
            values, mean, scale = standardize(x_enc, 1e-5, detach_mean=True, detach_std=True)
        else:
            values = x_enc
        hidden = self.input_projection(values)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.horizon_projection(
            self.final_norm(hidden).transpose(1, 2)
        ).transpose(1, 2)
        forecast = self.output_projection(hidden)
        return destandardize(forecast, mean, scale) if self.use_norm else forecast
