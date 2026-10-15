"""Independent implicit frequency-domain forecast decoder.

The implementation follows equations (1)--(6) of Li et al. (NeurIPS 2025): a
channel-separated encoder representation and the observed spectrum jointly
predict non-negative amplitudes plus continuous sine/cosine phase coordinates.
No source from the paper's reference repository was inspected or copied.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class Model(nn.Module):
    """Forecast by composing a learned pool of amplitude/phase wave models._components."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        frequency_pool: int | None = None,
        dropout: float = 0.0,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.frequency_pool = max(pred_len, frequency_pool or 2 * pred_len)
        self.pool_bins = self.frequency_pool // 2 + 1
        input_bins = seq_len // 2 + 1
        self.revin = RevIN(enc_in, enabled=use_revin)
        self.encoder = nn.Sequential(nn.Linear(seq_len, d_model), nn.GELU(), nn.Dropout(dropout))
        feature_width = d_model + input_bins
        self.amplitude_head = nn.Sequential(
            nn.Linear(feature_width, d_model), nn.GELU(), nn.Linear(d_model, self.pool_bins)
        )
        self.phase_sine_head = nn.Sequential(
            nn.Linear(feature_width, d_model), nn.GELU(), nn.Linear(d_model, self.pool_bins)
        )
        self.phase_cosine_head = nn.Sequential(
            nn.Linear(feature_width, d_model), nn.GELU(), nn.Linear(d_model, self.pool_bins)
        )

    def spectral_parameters(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return AHead/PHead outputs in the paper's non-negative/polar domains."""
        history = x.transpose(1, 2)
        spectrum = torch.fft.rfft(history, dim=-1)
        encoded = self.encoder(history)
        amplitude_features = torch.cat((encoded, spectrum.abs()), dim=-1)
        phase_features = torch.cat((encoded, torch.angle(spectrum)), dim=-1)
        amplitude = self.amplitude_head(amplitude_features).abs()
        sine = torch.tanh(self.phase_sine_head(phase_features))
        cosine = torch.tanh(self.phase_cosine_head(phase_features))
        phase = torch.atan2(sine, cosine).clamp(-math.pi, math.pi)
        return amplitude, phase

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        amplitude, phase = self.spectral_parameters(normalized)
        spectrum = torch.polar(amplitude, phase)
        signal = torch.fft.irfft(spectrum, n=self.frequency_pool, dim=-1)
        forecast = signal[..., : self.pred_len].transpose(1, 2)
        return self.revin(forecast, "denorm")
