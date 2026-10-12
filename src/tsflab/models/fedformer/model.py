"""Clean-room Fourier FEDformer forecast implementation from the ICML paper."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from tsflab.models._components.decomposition_encdec import DecompositionDecoderLayer, DecompositionEncoderLayer
from tsflab.models._components.forecast_embedding import ForecastEmbedding
from tsflab.models._components.series_decomposition import SeriesDecomposition
from tsflab.models._inputs import check_marks, check_window


def selected_modes(length: int, count: int, method: str) -> torch.Tensor:
    """Return deterministic low or paper-style random rFFT mode indices."""
    available = length // 2 + 1
    count = min(count, available)
    if method == "low":
        return torch.arange(count)
    generator = torch.Generator().manual_seed(104729 + 37 * length + count)
    return torch.randperm(available, generator=generator)[:count].sort().values


class FrequencyEnhancedBlock(nn.Module):
    """Paper equations (3)-(4), using a head-local complex spectral kernel."""

    def __init__(self, d_model: int, n_heads: int, length: int, modes: int, mode_select: str) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        indices = selected_modes(length, modes, mode_select)
        self.register_buffer("mode_indices", indices, persistent=True)
        count = len(indices)
        scale = 1.0 / max(1, self.head_dim)
        self.input_projection = nn.Linear(d_model, d_model)
        self.weight_real = nn.Parameter(
            scale * torch.randn(n_heads, count, self.head_dim, self.head_dim)
        )
        self.weight_imag = nn.Parameter(
            scale * torch.randn(n_heads, count, self.head_dim, self.head_dim)
        )
        self.output_projection = nn.Linear(d_model, d_model)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, width = values.shape
        projected = self.input_projection(values).view(
            batch, length, self.n_heads, self.head_dim
        )
        spectrum = torch.fft.rfft(projected, dim=1)
        indices = self.mode_indices[self.mode_indices < spectrum.shape[1]]
        selected = spectrum[:, indices]
        weight = torch.complex(
            self.weight_real[:, : len(indices)], self.weight_imag[:, : len(indices)]
        )
        transformed = torch.einsum("bmhi,hmio->bmho", selected, weight)
        output_spectrum = torch.zeros_like(spectrum)
        output_spectrum[:, indices] = transformed
        output = torch.fft.irfft(output_spectrum, n=length, dim=1)
        return self.output_projection(output.reshape(batch, length, width))


class FrequencyEnhancedAttention(nn.Module):
    """Paper equations (6)-(7): cross-attention in selected Fourier modes."""

    def __init__(self, d_model: int, n_heads: int, query_length: int, key_length: int, modes: int, mode_select: str) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.register_buffer(
            "query_modes", selected_modes(query_length, modes, mode_select), persistent=True
        )
        self.register_buffer(
            "key_modes", selected_modes(key_length, modes, mode_select), persistent=True
        )
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.last_attention: torch.Tensor | None = None

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        batch, query_length, width = query.shape
        key_length = context.shape[1]
        q = self.query(query).view(batch, query_length, self.n_heads, self.head_dim)
        k = self.key(context).view(batch, key_length, self.n_heads, self.head_dim)
        v = self.value(context).view(batch, key_length, self.n_heads, self.head_dim)
        q_frequency = torch.fft.rfft(q, dim=1)
        k_frequency = torch.fft.rfft(k, dim=1)
        v_frequency = torch.fft.rfft(v, dim=1)
        q_indices = self.query_modes[self.query_modes < q_frequency.shape[1]]
        k_indices = self.key_modes[self.key_modes < k_frequency.shape[1]]
        selected_q = q_frequency[:, q_indices]
        selected_k = k_frequency[:, k_indices]
        selected_v = v_frequency[:, k_indices]
        scores = torch.einsum(
            "bmhd,bnhd->bhmn", selected_q, torch.conj(selected_k)
        ).real / math.sqrt(self.head_dim)
        weights = torch.tanh(scores)
        mixed = torch.einsum(
            "bhmn,bnhd->bmhd", weights.to(selected_v.dtype), selected_v
        )
        mixed = mixed / max(1, len(k_indices))
        output_spectrum = torch.zeros_like(q_frequency)
        output_spectrum[:, q_indices] = mixed
        self.last_attention = weights.detach()
        output = torch.fft.irfft(output_spectrum, n=query_length, dim=1)
        return self.output(output.reshape(batch, query_length, width))


class FEDformerEncoderLayer(DecompositionEncoderLayer):
    def __init__(self, d_model: int, n_heads: int, length: int, modes: int, mode_select: str, d_ff: int, moving_avg: int, dropout: float, activation: str) -> None:
        super().__init__(
            FrequencyEnhancedBlock(d_model, n_heads, length, modes, mode_select),
            d_model, d_ff, moving_avg, dropout, activation, mixer_name="frequency_block",
        )


class FEDformerDecoderLayer(DecompositionDecoderLayer):
    def __init__(self, d_model: int, n_heads: int, query_length: int, key_length: int, modes: int, mode_select: str, d_ff: int, moving_avg: int, dropout: float, activation: str, c_out: int) -> None:
        super().__init__(
            FrequencyEnhancedBlock(d_model, n_heads, query_length, modes, mode_select),
            FrequencyEnhancedAttention(d_model, n_heads, query_length, key_length, modes, mode_select),
            d_model, d_ff, moving_avg, dropout, activation, c_out,
            mixer_names=("self_frequency", "cross_frequency"),
        )


class Model(nn.Module):
    """Forecast-only Fourier FEDformer with progressive decomposition."""

    def __init__(self, seq_len: int, label_len: int, pred_len: int, enc_in: int, dec_in: int, c_out: int, d_model: int, n_heads: int, e_layers: int, d_layers: int, d_ff: int, moving_avg: int, dropout: float, activation: str = "gelu", mode_select: str = "random", modes: int = 32) -> None:
        super().__init__()
        if label_len > seq_len:
            raise ValueError("label_len must not exceed seq_len")
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.channels = enc_in
        decoder_length = label_len + pred_len
        self.decomposition = SeriesDecomposition(moving_avg)
        self.encoder_embedding = ForecastEmbedding(enc_in, d_model, dropout)
        self.decoder_embedding = ForecastEmbedding(dec_in, d_model, dropout)
        self.encoder = nn.ModuleList([
            FEDformerEncoderLayer(d_model, n_heads, seq_len, modes, mode_select, d_ff, moving_avg, dropout, activation)
            for _ in range(e_layers)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.ModuleList([
            FEDformerDecoderLayer(d_model, n_heads, decoder_length, seq_len, modes, mode_select, d_ff, moving_avg, dropout, activation, c_out)
            for _ in range(d_layers)
        ])
        self.decoder_norm = nn.LayerNorm(d_model)
        self.seasonal_projection = nn.Linear(d_model, c_out)

    @staticmethod
    def _default_marks(values: torch.Tensor, length: int) -> torch.Tensor:
        return values.new_zeros(values.shape[0], length, 6)

    def _check_inputs(self, x_enc: torch.Tensor, x_dec: torch.Tensor | None) -> None:
        check_window(x_enc, self.seq_len, self.channels)
        # Same as ndim == 3, matching batch and shape[1:] == (label_len + pred_len, channels).
        if x_dec is not None and x_dec.shape != (x_enc.shape[0], self.label_len + self.pred_len, self.channels):
            raise ValueError("x_dec has an incompatible decoder contract")

    def _marks(self, x_enc: torch.Tensor, x_mark_enc, x_mark_dec) -> tuple[torch.Tensor, torch.Tensor]:
        """Encoder and decoder marks (zeros when absent), checked against the input and decoder lengths."""
        decoder_length = self.label_len + self.pred_len
        x_mark_enc = self._default_marks(x_enc, self.seq_len) if x_mark_enc is None else x_mark_enc
        x_mark_dec = self._default_marks(x_enc, decoder_length) if x_mark_dec is None else x_mark_dec
        check_marks(x_mark_enc, x_enc)
        if x_mark_dec.shape != (x_enc.shape[0], decoder_length, 6):
            raise ValueError("decoder marks must align with the decoder sequence")
        return x_mark_enc, x_mark_dec

    def _decoder_init(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Seasonal (label part, then zeros) and trend (label part, then the input mean) decoder inputs."""
        seasonal_history, trend_history = self.decomposition(x_enc)
        mean_future = x_enc.mean(dim=1, keepdim=True).expand(-1, self.pred_len, -1)
        zero_future = torch.zeros_like(mean_future)
        if self.label_len:
            seasonal = torch.cat((seasonal_history[:, -self.label_len :], zero_future), dim=1)
            trend = torch.cat((trend_history[:, -self.label_len :], mean_future), dim=1)
        else:
            seasonal, trend = zero_future, mean_future
        return seasonal, trend

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        self._check_inputs(x_enc, x_dec)
        x_mark_enc, x_mark_dec = self._marks(x_enc, x_mark_enc, x_mark_dec)
        seasonal, trend = self._decoder_init(x_enc)

        memory = self.encoder_embedding(x_enc, x_mark_enc)
        for layer in self.encoder:
            memory = layer(memory)
        memory = self.encoder_norm(memory)
        seasonal = self.decoder_embedding(seasonal, x_mark_dec)
        for layer in self.decoder:
            seasonal, trend = layer(seasonal, memory, trend)
        seasonal = self.seasonal_projection(self.decoder_norm(seasonal))
        return (seasonal + trend)[:, -self.pred_len :]
