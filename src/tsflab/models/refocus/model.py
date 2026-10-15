"""Paper-driven local implementation of ReFocus.

ReFocus reinforces mid-frequency and key-frequency modeling for multivariate
forecasting through two mechanisms:

* Adaptive Mid-Frequency Energy Optimizer (AMEO): a scaled moving-average
  residual (``x - beta * moving_average(x)``) that attenuates the very-low
  frequency (trend) band so mid-frequency spectral content is relatively
  reinforced before encoding.
* Energy-based Key-Frequency Picking Block (EKPB): a stack of encoder blocks
  that project features into the frequency domain, stochastically pool one
  channel's spectral content per frequency bin using a softmax energy
  distribution across channels (the shared "key frequency"), and fuse that
  pooled representation back with every channel's own features.

All frequency-domain linear projections (``FLinear`` in the official code) use
a dense complex matrix mapping every input frequency bin to every output
frequency bin (not a truncate/pad/interpolate scheme like FITS), so they
remain model-local.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tsflab.models._components.energy_frequency_pooling import EnergyBasedFrequencyPooling
from tsflab.models._components.revin import RevIN
from tsflab.models._components.series_decomposition import EdgePaddedMovingAverage
from tsflab.models._inputs import check_window


class FLinear(nn.Module):
    """Dense complex-linear projection between two sequence lengths via rFFT."""

    def __init__(self, inp: int, out: int) -> None:
        super().__init__()
        self.inp_size = inp // 2 + 1
        self.out_size = out // 2 + 1
        self.proj = nn.Linear(self.inp_size, self.out_size, dtype=torch.cfloat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.fft.irfft(self.proj(torch.fft.rfft(x, dim=-1)), n=self.out_size_time, dim=-1)

    @property
    def out_size_time(self) -> int:
        return (self.out_size - 1) * 2

    def initialize_as_identity(self) -> None:
        """Initialize the complex weight as a uniform averaging map (paper default)."""
        init_value = 1.0 / self.inp_size
        real_part = torch.full((self.out_size, self.inp_size), init_value)
        imaginary_part = torch.full((self.out_size, self.inp_size), init_value)
        with torch.no_grad():
            self.proj.weight.copy_(torch.complex(real_part, imaginary_part))


class EncoderBlock(nn.Module):
    """One Energy-based Key-Frequency Picking Block (EKPB)."""

    def __init__(self, d_model: int, d_pick: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = FLinear(d_model, d_model)
        self.fc2 = FLinear(d_model, d_pick)
        self.fc_core = FLinear(d_pick, d_model)
        self.fc_ori = FLinear(d_model, d_model)
        self.fc3 = FLinear(d_model, d_model)
        self.fc4 = FLinear(d_model, d_model)
        self.fc5 = FLinear(d_model, d_model)
        self.fc6 = FLinear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.pooling = EnergyBasedFrequencyPooling()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        core = F.gelu(self.fc1(inp))
        core = self.fc2(core)
        core_fft = torch.fft.rfft(core, dim=-1)
        pooled_fft = self.pooling(core_fft)
        core = torch.fft.irfft(pooled_fft.contiguous(), n=core.shape[-1], dim=-1)
        core = F.gelu(self.fc3(self.fc_core(core) + self.fc_ori(inp)))
        core = self.fc4(core)
        res = self.norm1(inp + self.dropout(core))
        hidden = self.dropout(F.gelu(self.fc5(res)))
        hidden = self.dropout(self.fc6(hidden))
        return self.norm2(res + hidden)


class Model(nn.Module):
    """RevIN -> mid-frequency reinforcement -> frequency embed -> EKPB stack -> projection."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        d_pick: int = 32,
        layers: int = 2,
        dropout: float = 0.1,
        beta: float = 0.5,
        kernel_size: int = 25,
        initial: bool = True,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 2:
            raise ValueError(
                "seq_len and pred_len must each be >= 2 (FLinear's irfft needs at least two output samples)"
            )
        if kernel_size > seq_len:
            raise ValueError("kernel_size must not exceed seq_len")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.beta = beta
        self.revin = RevIN(enc_in, affine=True)
        self.moving_average = EdgePaddedMovingAverage(kernel_size)
        self.embed = FLinear(seq_len, d_model)
        self.encoders = nn.ModuleList([EncoderBlock(d_model, d_pick, dropout) for _ in range(layers)])
        self.projection = FLinear(d_model, pred_len)
        if initial:
            self.projection.initialize_as_identity()

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        x_enc = self.revin(x_enc, "norm")
        x_enc = x_enc - self.beta * self.moving_average(x_enc)
        x_embed = self.embed(x_enc.transpose(1, 2))
        for encoder in self.encoders:
            x_embed = encoder(x_embed)
        pred = self.projection(x_embed).transpose(1, 2)
        return self.revin(pred, "denorm")
