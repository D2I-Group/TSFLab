"""Clean-room Dualformer forecast architecture.

Derived from the paper equations ("Dualformer: Time-Frequency Dual Domain
Learning for Long-term Time Series Forecasting", arXiv:2601.15669) and
clarified against the pinned official implementation
(https://github.com/Akira-221/Dualformer, revision
``ebd4ccf8bc5634f0c965d0b8d5797d1b926daa19``; see the model card for the
license and every recorded deviation). No source from that repository is
copied, vendored, or imported here.

Pipeline, per the official code's ``Model.forward``:

1. RevIN-normalize the raw history, then embed values and calendar marks.
2. At each of ``e_layers`` depths, take the running time-branch and
   frequency-branch states, FFT them, select a depth-indexed contiguous
   frequency band with :class:`HierarchicalFrequencySampler` (paper's
   Hierarchical Frequency Sampling / HFS module: shallow layers keep
   high-frequency detail, deep layers specialize on near-DC trend), and
   invert the zero-padded band back to a length-``seq_len`` real signal.
2. Feed that band-limited signal through one Transformer encoder layer per
   branch: the time branch uses ordinary full self-attention, the frequency
   branch uses a period-matching auto-correlation attention (paper's
   frequency branch, following the Wiener-Khinchin autocorrelation
   identity used by Autoformer-style frequency attention).
3. Fuse the two branches' final states with a periodicity-aware gate
   (:class:`HarmonicEnergyGate`, the paper's harmonic-energy weighting) and
   project only the last fused time step to the forecast horizon, then
   RevIN-denormalize.

Deviation from the pinned official code (recorded in the model card): the
reference implementation recomputes every layer's band directly from the
*original* embedding's spectrum and discards each layer's encoder output
except the last, so only the final encoder layer of each branch ever
contributes to the forecast and to gradients. That silently collapses the
paper's stated depth-wise curriculum ("shallow layers keep high-frequency
detail, deep layers model low-frequency trend") into a single active layer.
This implementation instead re-derives the spectrum from each branch's
*running* state every layer, so all ``e_layers`` layers are genuinely
stacked and contribute to the output, matching the paper's stated intent.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._components.forecast_embedding import ForecastEmbedding
from tsflab.models._components.frequency_band_sampler import HierarchicalFrequencySampler
from tsflab.models._components.harmonic_energy_gate import HarmonicEnergyGate
from tsflab.models._components.revin import RevIN
from tsflab.models._components.self_attention_family import AttentionLayer, FullAttention
from tsflab.models._components.transformer_encdec import EncoderLayer
from tsflab.models._inputs import check_window


class AutoCorrelationAttention(nn.Module):
    """Model-local frequency-branch attention: the paper's period-based
    auto-correlation mechanism, reimplemented clean-room with the
    ``(queries, keys, values, attn_mask, tau, delta) -> (out, attn)`` contract
    so it slots into the shared :class:`AttentionLayer`/:class:`EncoderLayer`
    components exactly like :class:`FullAttention`. Not extracted as a shared
    component: it is a defining, paper-specific attention rule rather than a
    generic building block.
    """

    def __init__(self, factor: float, dropout: float) -> None:
        super().__init__()
        self.factor = factor
        self.dropout = nn.Dropout(dropout)
        self.last_delays: torch.Tensor | None = None

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        batch, length, heads, dim = queries.shape
        _, key_length, _, _ = keys.shape
        if key_length > length:
            keys = keys[:, :length]
            values = values[:, :length]
        elif key_length < length:
            pad = queries.new_zeros(batch, length - key_length, heads, dim)
            keys = torch.cat([keys, pad], dim=1)
            values = torch.cat([values, pad], dim=1)

        # Wiener-Khinchin: correlation via FFT along the time axis, per head/dim.
        q = queries.permute(0, 2, 3, 1).contiguous()  # (B, H, D, L)
        k = keys.permute(0, 2, 3, 1).contiguous()
        q_freq = torch.fft.rfft(q, dim=-1)
        k_freq = torch.fft.rfft(k, dim=-1)
        correlation = torch.fft.irfft(q_freq * torch.conj(k_freq), n=length, dim=-1)

        mean_correlation = correlation.mean(dim=(1, 2))  # (B, L)
        top_k = max(1, min(length, int(self.factor * math.log(max(length, 2)))))
        scores, delays = torch.topk(mean_correlation, top_k, dim=-1)
        weights = torch.softmax(scores, dim=-1)

        v = values.permute(0, 2, 3, 1).contiguous()  # (B, H, D, L)
        aggregated = torch.zeros_like(v)
        for rank in range(top_k):
            rolled = torch.stack(
                [torch.roll(v[sample], shifts=-int(delays[sample, rank]), dims=-1) for sample in range(batch)],
                dim=0,
            )
            aggregated = aggregated + rolled * weights[:, rank].view(batch, 1, 1, 1)
        self.last_delays = delays.detach()
        out = self.dropout(aggregated).permute(0, 3, 1, 2)  # (B, L, H, D)
        return out.contiguous(), None


class Model(nn.Module):
    """Dual time/frequency-domain Transformer with hierarchical frequency
    sampling and periodicity-aware branch fusion."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        c_out: int,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 3,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        factor: float = 1.0,
        alpha: float = 1.0,
        num_harmonics: int = 3,
        dc_bins: int = 3,
    ) -> None:
        super().__init__()

        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.c_out = c_out
        self.n_layers = e_layers

        self.revin = RevIN(enc_in, affine=True)
        self.embedding = ForecastEmbedding(enc_in, d_model, dropout)
        self.freq_sampler = HierarchicalFrequencySampler(e_layers, alpha=alpha)
        self.gate = HarmonicEnergyGate(num_harmonics=num_harmonics, low_freq_guard=dc_bins)

        self.time_layers = nn.ModuleList(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout, output_attention=False),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ]
        )
        self.freq_layers = nn.ModuleList(
            [
                EncoderLayer(
                    AttentionLayer(AutoCorrelationAttention(factor, dropout), d_model, n_heads),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ]
        )
        self.time_norm = nn.LayerNorm(d_model)
        self.freq_norm = nn.LayerNorm(d_model)
        self.projection = nn.Linear(d_model, pred_len * c_out)

        self.last_bands: list[tuple[int, int]] | None = None
        self.last_gate_weight: torch.Tensor | None = None

    @staticmethod
    def _default_marks(values: torch.Tensor, length: int) -> torch.Tensor:
        return values.new_zeros(values.shape[0], length, 6)

    def _band_limited_signal(self, state: torch.Tensor, layer_idx: int) -> tuple[torch.Tensor, tuple[int, int]]:
        """FFT ``state``, keep only layer ``layer_idx``'s HFS band, and invert."""
        spectrum = torch.fft.rfft(state, dim=1)
        start, end = self.freq_sampler.band(spectrum.shape[1], layer_idx)
        band_limited = torch.zeros_like(spectrum)
        band_limited[:, start:end] = spectrum[:, start:end]
        signal = torch.fft.irfft(band_limited, n=self.seq_len, dim=1)
        return signal, (start, end)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        check_window(x_enc, self.seq_len, self.channels)
        x_mark_enc = self._default_marks(x_enc, self.seq_len) if x_mark_enc is None else x_mark_enc
        if x_mark_enc.shape[:2] != x_enc.shape[:2]:
            raise ValueError("encoder marks do not align with x_enc")

        normalized = self.revin(x_enc, "norm")
        embedded = self.embedding(normalized, x_mark_enc)

        time_state, freq_state = embedded, embedded
        bands: list[tuple[int, int]] = []
        for layer_idx in range(self.n_layers):
            time_input, bound = self._band_limited_signal(time_state, layer_idx)
            time_state, _ = self.time_layers[layer_idx](time_input, attn_mask=None)

            freq_input, _ = self._band_limited_signal(freq_state, layer_idx)
            freq_state, _ = self.freq_layers[layer_idx](freq_input, attn_mask=None)
            bands.append(bound)
        self.last_bands = bands

        time_state = self.time_norm(time_state)
        freq_state = self.freq_norm(freq_state)

        freq_weight = self.gate(embedded)  # (B, 1, d_model), periodicity-aware gate
        self.last_gate_weight = freq_weight.detach()
        fused = freq_state * freq_weight + time_state * (1.0 - freq_weight)

        last_step = fused[:, -1, :]
        forecast = self.projection(last_step).view(x_enc.shape[0], self.pred_len, self.c_out)
        return self.revin(forecast, "denorm")
