"""Paper-driven local implementation of FilterNet's contextual shaping filter."""

from __future__ import annotations

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN


def _complex_linear(values: torch.Tensor, real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    weight = torch.complex(real, imag)
    return torch.einsum("bcf,fd->bcd", values, weight)


class ContextualShapingFilter(nn.Module):
    """Embed F(Z), generate H_phi(F(Z)), then filter the embedded spectrum."""

    def __init__(self, input_length: int, output_length: int) -> None:
        super().__init__()
        input_bins = input_length // 2 + 1
        output_bins = output_length // 2 + 1
        self.output_length = output_length
        scale = input_bins**-0.5
        self.embed_real = nn.Parameter(torch.randn(input_bins, output_bins) * scale)
        self.embed_imag = nn.Parameter(torch.randn(input_bins, output_bins) * scale)
        self.context_real = nn.Parameter(torch.ones(2, output_bins))
        self.context_imag = nn.Parameter(torch.zeros(2, output_bins))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(values, dim=1).transpose(1, 2)
        embedded = _complex_linear(spectrum, self.embed_real, self.embed_imag)
        context = embedded
        for real, imag in zip(self.context_real, self.context_imag):
            product = context * torch.complex(real, imag).view(1, 1, -1)
            context = torch.complex(torch.relu(product.real), torch.relu(product.imag))
        filtered = embedded * context
        return torch.fft.irfft(filtered, n=self.output_length, dim=-1).transpose(1, 2)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        embed_size: int = 128,
        hidden_size: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.normalization = RevIN(enc_in)
        self.filter = ContextualShapingFilter(seq_len, embed_size)
        self.forecast = nn.Sequential(
            nn.Linear(embed_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, pred_len),
        )

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
