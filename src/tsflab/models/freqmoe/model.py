"""Paper-driven local implementation of FreqMoE.

FreqMoE first passes the input window through a Frequency-band Decomposition
Mixture of Experts (`FrequencyBandMixtureOfExperts`, shared component) that
reconstructs a denoised version of the series from a learned, input-gated
mixture of contiguous frequency bands. It then predicts the horizon with a
small stack of residual frequency-extension blocks: each block upsamples the
rFFT spectrum from `seq_len` to `seq_len + pred_len` bins with a complex
linear layer, applies a complex ReLU and complex dropout, refines with a
second complex linear layer, and reconstructs via `irfft`; blocks are chained
on the residual left after subtracting the previous blocks' backcast, and
their forecast segments are summed into the final prediction.
"""

from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.freq_band_moe import FrequencyBandMixtureOfExperts


class ComplexReLU(nn.Module):
    """Apply ReLU independently to the real and imaginary parts."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(torch.relu(x.real), torch.relu(x.imag))


class ComplexDropout(nn.Module):
    """Apply the same-rate dropout independently to real and imaginary parts."""

    def __init__(self, dropout_rate: float = 0.3) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.complex(self.dropout(x.real), self.dropout(x.imag))


class FrequencyExtensionBlock(nn.Module):
    """Instance-normalize, extend the spectrum to seq_len + pred_len, and invert."""

    def __init__(self, seq_len: int, pred_len: int, dropout: float) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.length_ratio = (seq_len + pred_len) / seq_len
        dominance_freq = seq_len // 2 + 1
        extended_freq = (seq_len + pred_len) // 2 + 1
        self.freq_upsampler = nn.Linear(dominance_freq, extended_freq, dtype=torch.cfloat)
        self.freq_upsampler1 = nn.Linear(extended_freq, extended_freq, dtype=torch.cfloat)
        self.complex_relu = ComplexReLU()
        self.complex_dropout = ComplexDropout(dropout_rate=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = torch.mean(x, dim=1, keepdim=True)
        centered = x - x_mean
        x_var = torch.var(centered, dim=1, keepdim=True) + 1e-5
        normalized = centered / torch.sqrt(x_var)

        spectrum = torch.fft.rfft(normalized, dim=1)
        spectrum = self.freq_upsampler(spectrum.permute(0, 2, 1)).permute(0, 2, 1)
        spectrum = self.complex_relu(spectrum)
        spectrum = self.complex_dropout(spectrum)
        spectrum = self.freq_upsampler1(spectrum.permute(0, 2, 1)).permute(0, 2, 1)
        extended = torch.fft.irfft(spectrum, n=self.seq_len + self.pred_len, dim=1)
        extended = extended * self.length_ratio
        return extended * torch.sqrt(x_var) + x_mean


class Model(nn.Module):
    """Frequency-band MoE denoiser followed by residual frequency-extension blocks."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        expert_num: int = 4,
        freq_num_blocks: int = 1,
        dropout_freq: float = 0.1,
    ) -> None:
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("seq_len and pred_len must be positive")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.moe = FrequencyBandMixtureOfExperts(expert_num, seq_len)
        self.blocks = nn.ModuleList(
            [FrequencyExtensionBlock(seq_len, pred_len, dropout_freq) for _ in range(freq_num_blocks)]
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError("x_enc does not match configured time/channel dimensions")
        denoised, self.last_band_boundaries, self.last_gating_scores = self.moe(
            x_enc.permute(0, 2, 1)
        )
        denoised = denoised.permute(0, 2, 1)

        residual = denoised
        total_prediction = x_enc.new_zeros(x_enc.shape[0], self.pred_len, self.enc_in)
        for index, block in enumerate(self.blocks):
            prediction = block(residual)
            if index == 0:
                residual = denoised - prediction[:, : self.seq_len]
            else:
                residual = residual - prediction[:, : self.seq_len]
            total_prediction = total_prediction + prediction[:, self.seq_len :]
        return total_prediction
