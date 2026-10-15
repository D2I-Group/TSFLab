"""Clean-room Adaptive Wavelet-Enhanced Mixer (arXiv:2511.04722)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tsflab.models._components.revin import RevIN
from tsflab.models._components.wavelet import UndecimatedWaveletTransform
from tsflab.models._inputs import check_window


class MultiScaleTemporalEmbedding(nn.Module):
    """Dilated-receptive-field "anchor" embeddings of the raw sequence.

    For scale ``s`` in ``1..num_scales``, a same-padded ``Conv1d`` with kernel
    ``2s + 1`` followed by global average pooling produces one time-domain
    summary vector per scale; stacking over scales gives a
    ``(batch*channels, num_scales, d_model)`` tensor.
    """

    def __init__(self, num_scales: int, d_model: int, hidden: int = 16) -> None:
        super().__init__()
        self.branches = nn.ModuleList()
        for scale in range(1, num_scales + 1):
            kernel_size = 2 * scale + 1
            padding = kernel_size // 2
            self.branches.append(
                nn.Sequential(
                    nn.ReflectionPad1d(padding),
                    nn.Conv1d(1, hidden, kernel_size=kernel_size),
                    nn.GELU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(hidden, d_model),
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch*channels, length)
        signal = x.unsqueeze(1)
        return torch.stack([branch(signal) for branch in self.branches], dim=1)


class FrequencyRouter(nn.Module):
    """Adaptively weight wavelet subbands using four hand-crafted descriptors.

    Global FFT band energy, per-band wavelet energy, the strongest local
    (sliding-window) burst energy, and per-band spectral entropy are each
    z-score normalized across subbands, concatenated, and mapped by a small
    MLP to a softmax weighting over the ``num_bands`` subbands.
    """

    def __init__(self, num_bands: int, hidden: int = 64) -> None:
        super().__init__()
        self.num_bands = num_bands
        self.router = nn.Sequential(nn.Linear(4 * num_bands, hidden), nn.GELU(), nn.Linear(hidden, num_bands))

    @staticmethod
    def _zscore_across_bands(values: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=1, keepdim=True)
        std = values.std(dim=1, keepdim=True, unbiased=False) + 1e-8
        return (values - mean) / std

    def forward(self, signal: torch.Tensor, subbands: torch.Tensor) -> torch.Tensor:
        # signal: (batch*channels, length); subbands: (batch*channels, num_bands, length)
        batch_channels, num_bands, length = subbands.shape

        amplitude = torch.fft.rfft(signal, dim=-1).abs()
        num_freqs = length // 2 + 1
        total_energy = (amplitude**2).sum(-1, keepdim=True) + 1e-8
        band_edges = torch.linspace(0, num_freqs, num_bands + 1).round().long()
        fft_energy = torch.stack(
            [(amplitude[:, band_edges[j] : band_edges[j + 1]] ** 2).sum(-1) for j in range(num_bands)],
            dim=1,
        ) / total_energy.squeeze(-1).unsqueeze(1)

        wavelet_energy = (subbands**2).mean(-1)

        window = min(16, length)
        local_energy = F.avg_pool1d(subbands**2, kernel_size=window, stride=1).amax(-1)

        power = subbands**2
        prob = power / (power.sum(-1, keepdim=True) + 1e-8)
        entropy = -(1.0 / torch.log(torch.tensor(float(length)))) * (prob * (prob + 1e-8).log()).sum(-1)

        descriptors = torch.cat(
            [
                self._zscore_across_bands(fft_energy),
                self._zscore_across_bands(wavelet_energy),
                self._zscore_across_bands(local_energy),
                self._zscore_across_bands(entropy),
            ],
            dim=-1,
        )
        return self.router(descriptors).softmax(dim=-1).unsqueeze(-1)


class CoherentGatedFusion(nn.Module):
    """Cross-attend temporal anchors to weighted wavelet features, then gate.

    A single-head cross-attention lets each temporal-scale anchor query the
    frequency features; a sigmoid gate conditioned on both decides how much
    of that frequency context to inject residually, followed by LayerNorm.
    """

    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = d_model**-0.5
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())
        self.norm = nn.LayerNorm(d_model)

    def forward(self, temporal: torch.Tensor, frequency: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = self.query_proj(temporal), self.key_proj(frequency), self.value_proj(frequency)
        attn = self.dropout((q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1))
        context = self.out_proj(attn @ v)
        gate = self.gate(torch.cat([temporal, context], dim=-1))
        fused = temporal + gate * context
        return self.norm(fused), gate


class CrossScaleMixer(nn.Module):
    """Mix representations across the temporal-scale axis with a small MLP."""

    def __init__(self, num_scales: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mixer = nn.Sequential(
            nn.Linear(num_scales, num_scales * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_scales * 2, num_scales),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        mixed = self.mixer(self.norm(x).transpose(1, 2)).transpose(1, 2)
        return residual + mixed


class Model(nn.Module):
    """AWEMixer: a Frequency Router adaptively weights undecimated wavelet
    subbands using global-periodicity (FFT) evidence, and a Coherent Gated
    Fusion Block lets multi-scale temporal anchors selectively absorb that
    weighted frequency context through cross-attention and a sigmoid gate.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        dropout: float = 0.2,
        num_scales: int = 3,
        wavelet_level: int = 3,
        wavelet: str = "db4",
        num_fusion_layers: int = 1,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.num_bands = wavelet_level + 1

        self.revin = RevIN(enc_in)
        self.temporal_embedding = MultiScaleTemporalEmbedding(num_scales, d_model)
        self.wavelet = UndecimatedWaveletTransform(wavelet, wavelet_level)
        self.band_embedding = nn.ModuleList(nn.Linear(seq_len, d_model) for _ in range(self.num_bands))
        self.router = FrequencyRouter(self.num_bands)
        self.fusion_layers = nn.ModuleList(CoherentGatedFusion(d_model, dropout) for _ in range(num_fusion_layers))
        self.cross_scale_mixer = CrossScaleMixer(num_scales, d_model, dropout)
        self.head = nn.Linear(d_model, pred_len)

        self.last_router_weights: torch.Tensor | None = None
        self.last_gates: list[torch.Tensor] = []

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        check_window(x_enc, self.seq_len, self.enc_in)
        batch = x_enc.shape[0]
        normalized = self.revin(x_enc, "norm")
        flat = normalized.transpose(1, 2).reshape(batch * self.enc_in, self.seq_len)

        temporal = self.temporal_embedding(flat)

        subbands = [band.squeeze(1) for band in self.wavelet(flat.unsqueeze(1))]
        subband_stack = torch.stack(subbands, dim=1)
        frequency = torch.stack([embed(subbands[i]) for i, embed in enumerate(self.band_embedding)], dim=1)

        weights = self.router(flat, subband_stack)
        self.last_router_weights = weights
        weighted_frequency = frequency * weights

        fused = temporal
        gates = []
        for layer in self.fusion_layers:
            fused, gate = layer(fused, weighted_frequency)
            gates.append(gate)
        self.last_gates = gates

        mixed = self.cross_scale_mixer(fused).mean(dim=1)
        forecast = self.head(mixed).reshape(batch, self.enc_in, self.pred_len).transpose(1, 2)
        return self.revin(forecast, "denorm")
