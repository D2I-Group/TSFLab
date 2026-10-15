"""Clean-room MixLinear implementation from the ICLR 2026 paper."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._inputs import check_window


class SegmentTrendPath(nn.Module):
    """Factor local segment shapes and dependencies between segments."""

    def __init__(self, effective_length: int, segments: int, hidden_rank: int) -> None:
        super().__init__()
        if effective_length % segments:
            raise ValueError("downsampled length must be divisible by segments")
        self.segments = segments
        self.segment_length = effective_length // segments
        self.local_encoder = nn.Linear(self.segment_length, hidden_rank)
        self.segment_mixer = nn.Linear(segments, segments)
        self.local_decoder = nn.Linear(hidden_rank, self.segment_length)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = x.shape
        pieces = x.reshape(batch, channels, self.segments, self.segment_length)
        local = self.local_encoder(pieces)
        related = self.segment_mixer(local.transpose(-1, -2)).transpose(-1, -2)
        return self.local_decoder(related).reshape(batch, channels, -1)


class LowRankSpectralPath(nn.Module):
    """Equation (4): a complex rank-constrained operator U(VF)."""

    def __init__(self, effective_length: int, rank: int) -> None:
        super().__init__()
        if rank > effective_length:
            raise ValueError("spectral_rank cannot exceed downsampled length")
        scale = effective_length**-0.5
        self.analysis_real = nn.Parameter(torch.randn(rank, effective_length) * scale)
        self.analysis_imag = nn.Parameter(torch.randn(rank, effective_length) * scale)
        self.synthesis_real = nn.Parameter(torch.randn(effective_length, rank) * scale)
        self.synthesis_imag = nn.Parameter(torch.randn(effective_length, rank) * scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spectrum = torch.fft.fft(x, dim=-1)
        analysis = torch.complex(self.analysis_real, self.analysis_imag)
        synthesis = torch.complex(self.synthesis_real, self.synthesis_imag)
        latent = torch.einsum("rn,bcn->bcr", analysis, spectrum)
        filtered = torch.einsum("nr,bcr->bcn", synthesis, latent)
        return torch.fft.ifft(filtered, dim=-1).real


class Model(nn.Module):
    """Add segment-domain and frequency-domain forecasts as in Equation (1)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        downsample: int = 4,
        segments: int = 4,
        hidden_rank: int = 2,
        spectral_rank: int = 2,
    ) -> None:
        super().__init__()
        if seq_len % downsample:
            raise ValueError("seq_len must be divisible by downsample")
        effective_length = seq_len // downsample
        if effective_length % segments:
            raise ValueError("seq_len/downsample must be divisible by segments")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.downsample = downsample
        self.segment_path = SegmentTrendPath(effective_length, segments, hidden_rank)
        self.spectral_path = LowRankSpectralPath(effective_length, spectral_rank)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.enc_in)
        center = x_enc.mean(dim=1, keepdim=True)
        history = (x_enc - center).transpose(1, 2)
        reduced = F.avg_pool1d(history, kernel_size=self.downsample, stride=self.downsample)
        segment = self.segment_path(reduced)
        frequency = self.spectral_path(reduced)
        combined = F.interpolate(
            segment + frequency,
            size=self.pred_len,
            mode="linear",
            align_corners=False,
        )
        return combined.transpose(1, 2) + center
