"""Paper-driven local implementation of FilterNet's plain shaping filter."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN


class PlainShapingFilter(nn.Module):
    """S = F^-1(F(Z) elementwise-multiplied by H_phi)."""

    def __init__(self, length: int) -> None:
        super().__init__()
        bins = length // 2 + 1
        self.length = length
        self.weight_real = nn.Parameter(torch.ones(bins))
        self.weight_imag = nn.Parameter(torch.zeros(bins))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(values, dim=1)
        weight = torch.complex(self.weight_real, self.weight_imag).view(1, -1, 1)
        return torch.fft.irfft(spectrum * weight, n=self.length, dim=1)


class Model(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, enc_in: int, hidden_size: int = 256) -> None:
        super().__init__()
        self.normalization = RevIN(enc_in)
        self.filter = PlainShapingFilter(seq_len)
        self.forecast = nn.Sequential(nn.Linear(seq_len, hidden_size), nn.GELU(), nn.Linear(hidden_size, pred_len))

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        normalized = self.normalization(x_enc, "norm")
        filtered = self.filter(normalized)
        prediction = self.forecast(filtered.transpose(1, 2)).transpose(1, 2)
        return self.normalization(prediction, "denorm")
