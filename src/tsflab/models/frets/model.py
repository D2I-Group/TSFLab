"""Clean-room FreTS implementation from Eqs. 1--7 of the NeurIPS paper."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._inputs import check_window


class ComplexFrequencyMLP(nn.Module):
    """Full complex matrix mapping expanded into real arithmetic (Eq. 7)."""

    def __init__(self, width: int, sparsity_threshold: float) -> None:
        super().__init__()
        scale = width**-0.5
        self.real_weight = nn.Parameter(torch.randn(width, width) * scale)
        self.imag_weight = nn.Parameter(torch.randn(width, width) * scale)
        self.real_bias = nn.Parameter(torch.zeros(width))
        self.imag_bias = nn.Parameter(torch.zeros(width))
        self.sparsity_threshold = sparsity_threshold

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        real = F.linear(values.real, self.real_weight, self.real_bias)
        real = real - F.linear(values.imag, self.imag_weight, None)
        imag = F.linear(values.real, self.imag_weight, self.imag_bias)
        imag = imag + F.linear(values.imag, self.real_weight, None)
        real, imag = F.relu(real), F.relu(imag)
        if self.sparsity_threshold:
            real = F.softshrink(real, self.sparsity_threshold)
            imag = F.softshrink(imag, self.sparsity_threshold)
        return torch.complex(real, imag)


class FrequencyChannelLearner(nn.Module):
    """Inter-series dependency learner from paper Eq. 3."""

    def __init__(self, width: int, threshold: float) -> None:
        super().__init__()
        self.mlp = ComplexFrequencyMLP(width, threshold)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(hidden.transpose(1, 2), dim=2, norm="ortho")
        learned = self.mlp(spectrum)
        channels = hidden.shape[1]
        return torch.fft.irfft(learned, n=channels, dim=2, norm="ortho").transpose(1, 2)


class FrequencyTemporalLearner(nn.Module):
    """Intra-series temporal dependency learner from paper Eq. 4."""

    def __init__(self, width: int, threshold: float) -> None:
        super().__init__()
        self.mlp = ComplexFrequencyMLP(width, threshold)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.rfft(hidden, dim=2, norm="ortho")
        learned = self.mlp(spectrum)
        return torch.fft.irfft(learned, n=hidden.shape[2], dim=2, norm="ortho")


class Model(nn.Module):
    """FreTS domain conversion, two frequency learners, and direct FFN head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        features: str = "M",
        embed_size: int = 128,
        hidden_size: int = 256,
        channel_independence: bool = False,
        sparsity_threshold: float = 0.01,
    ) -> None:
        super().__init__()
        del features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.dimension_extension = nn.Parameter(torch.randn(embed_size) * 0.02)
        # With one or two real-valued channels every rFFT bin is real, so the
        # complex channel map is degenerate and is intentionally not created.
        self.channel_learner = (
            None
            if channel_independence or enc_in < 3
            else FrequencyChannelLearner(embed_size, sparsity_threshold)
        )
        self.temporal_learner = FrequencyTemporalLearner(embed_size, sparsity_threshold)
        self.forecast_head = nn.Sequential(
            nn.Linear(seq_len * embed_size, hidden_size),
            nn.LeakyReLU(),
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
        check_window(x_enc, self.seq_len, self.channels)
        hidden = x_enc.transpose(1, 2).unsqueeze(-1) * self.dimension_extension
        residual = hidden
        if self.channel_learner is not None:
            hidden = self.channel_learner(hidden)
        hidden = self.temporal_learner(hidden) + residual
        forecast = self.forecast_head(hidden.flatten(2))
        return forecast.transpose(1, 2)
