"""Clean-room Fredformer based only on the KDD paper.

The forecast path transforms each variable to the frequency domain, forms
contiguous bands, removes per-band energy imbalance, and applies the same
channel-wise Transformer to every band before reconstructing a history signal.
No source from the unlicensed reference repository is copied or reused here.
"""
from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class FrequencyEqualization(nn.Module):
    """Normalize each complex frequency band to unit RMS energy."""
    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, bands: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        energy = bands.abs().square().mean((-1, -2), keepdim=True).add(self.eps).sqrt()
        return bands / energy, energy


class FrequencyBandAttention(nn.Module):
    """Shared channel-wise self-attention independently applied per band."""
    def __init__(self, band_width: int, model_width: int, depth: int, heads: int,
                 feedforward: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(2 * band_width, model_width)
        layer = nn.TransformerEncoderLayer(
            model_width, heads, feedforward, dropout, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, depth)
        self.output = nn.Linear(model_width, 2 * band_width)

    def forward(self, bands: torch.Tensor) -> torch.Tensor:
        batch, count, channels, width = bands.shape
        features = torch.cat((bands.real, bands.imag), -1).reshape(batch * count, channels, 2 * width)
        encoded = self.output(self.encoder(self.input(features))).reshape(batch, count, channels, 2 * width)
        real, imag = encoded.chunk(2, -1)
        return torch.complex(real, imag)


def split_frequency_bands(spectrum: torch.Tensor, band_width: int) -> tuple[torch.Tensor, int]:
    """Pad and split ``(B,C,F)`` complex spectra into contiguous bands."""
    bins = spectrum.shape[-1]
    padded_bins = ((bins + band_width - 1) // band_width) * band_width
    if padded_bins != bins:
        spectrum = torch.nn.functional.pad(spectrum, (0, padded_bins - bins))
    bands = spectrum.unfold(-1, band_width, band_width).permute(0, 2, 1, 3)
    return bands, bins


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, label_len=0, features="M",
                 band_width=16, model_width=48, depth=2, heads=6,
                 feedforward=128, dropout=0.2, revin=True, affine=True,
                 subtract_last=False, head_dropout=0.0):
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.band_width = band_width
        self.revin = RevIN(enc_in, affine=affine, subtract_last=subtract_last, enabled=revin)
        self.equalize = FrequencyEqualization()
        self.band_transformer = FrequencyBandAttention(
            band_width, model_width, depth, heads, feedforward, dropout
        )
        self.band_gate = nn.Sequential(nn.Linear(2 * band_width, model_width), nn.GELU(), nn.Linear(model_width, 1))
        self.head = nn.Sequential(nn.Dropout(head_dropout), nn.Linear(seq_len, pred_len))

    def frequency_debias(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum = torch.fft.rfft(values.transpose(1, 2), dim=-1)
        bands, bins = split_frequency_bands(spectrum, self.band_width)
        normalized, energy = self.equalize(bands)
        transformed = self.band_transformer(normalized)
        features = torch.cat((normalized.real, normalized.imag), -1).mean(2)
        gate = torch.sigmoid(self.band_gate(features)).unsqueeze(-1)
        debiased = (normalized + gate * transformed) * energy
        merged = debiased.permute(0, 2, 1, 3).flatten(-2)[..., :bins]
        return torch.fft.irfft(merged, n=self.seq_len, dim=-1), energy.squeeze(-1).squeeze(-1)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        reconstructed, _ = self.frequency_debias(normalized)
        forecast = self.head(reconstructed).transpose(1, 2)
        return self.revin(forecast, "denorm")
