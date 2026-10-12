"""Clean-room TimeEmb static/dynamic disentanglement implementation."""
from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.revin import RevIN


class GlobalCalendarEmbedding(nn.Module):
    """Persistent full-spectrum representations indexed by forecast calendar."""
    def __init__(self, slots, channels, bins):
        super().__init__()
        self.slots = slots
        self.real = nn.Parameter(torch.zeros(slots, channels, bins))
        self.imag = nn.Parameter(torch.zeros(slots, channels, bins))
        nn.init.normal_(self.real, std=0.02)
        nn.init.normal_(self.imag, std=0.02)

    def forward(self, indices):
        indices = indices.remainder(self.slots)
        return torch.complex(self.real[indices], self.imag[indices])


class DynamicSpectrumFilter(nn.Module):
    """Efficient full-spectrum filter conditioned on the current spectrum."""
    def __init__(self, bins, hidden, scale):
        super().__init__()
        self.response_real = nn.Parameter(torch.ones(bins))
        self.response_imag = nn.Parameter(torch.randn(bins) * scale)
        self.conditioner = nn.Sequential(nn.Linear(bins, hidden), nn.GELU(), nn.Linear(hidden, bins), nn.Sigmoid())

    def forward(self, spectrum):
        energy = spectrum.abs().mean(1)
        gate = self.conditioner(energy)
        response = torch.complex(self.response_real, self.response_imag)
        return spectrum * (1 + gate.unsqueeze(1) * response)


class Model(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, d_model=128, use_revin=True,
                 use_hour_index=True, use_day_index=False, scale=0.02,
                 hour_length=24, day_length=7):
        super().__init__()
        if min(seq_len, pred_len) < 1:
            raise ValueError("invalid non-positive TimeEmb dimension")
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.use_hour_index, self.use_day_index = use_hour_index, use_day_index
        bins = seq_len // 2 + 1
        self.revin = RevIN(enc_in, enabled=use_revin)
        self.hour_embedding = GlobalCalendarEmbedding(hour_length, enc_in, bins) if use_hour_index else None
        self.day_embedding = GlobalCalendarEmbedding(day_length, enc_in, bins) if use_day_index else None
        self.dynamic_filter = DynamicSpectrumFilter(bins, d_model, scale)
        self.forecast = nn.Sequential(nn.Linear(seq_len, d_model), nn.GELU(), nn.Linear(d_model, pred_len))
        self.last_static_spectrum = None

    def _indices(self, x, x_mark_enc, x_mark_dec):
        marks = x_mark_dec[:, -self.pred_len] if x_mark_dec is not None else (x_mark_enc[:, -1] if x_mark_enc is not None else None)
        if marks is None:
            zero = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            return zero, zero
        if marks.shape[-1] >= 6:
            return marks[:, 4].long(), marks[:, 3].long()
        hour = marks[:, -1].long()
        day = marks[:, -2].long() if marks.shape[-1] > 1 else hour.new_zeros(hour.shape)
        return hour, day

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        if x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(f"expected (*,{self.seq_len},{self.enc_in})")
        normalized = self.revin(x_enc, "norm").transpose(1, 2)
        spectrum = torch.fft.rfft(normalized, dim=-1, norm="ortho")
        hour, day = self._indices(x_enc, x_mark_enc, x_mark_dec)
        static = torch.zeros_like(spectrum)
        if self.hour_embedding is not None:
            static = static + self.hour_embedding(hour)
        if self.day_embedding is not None:
            static = static + self.day_embedding(day)
        self.last_static_spectrum = static
        dynamic = self.dynamic_filter(spectrum - static)
        restored = torch.fft.irfft(dynamic + static, n=self.seq_len, dim=-1, norm="ortho")
        return self.revin(self.forecast(restored).transpose(1, 2), "denorm")
