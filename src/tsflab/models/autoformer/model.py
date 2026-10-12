"""Clean-room Autoformer forecast architecture derived from the paper equations."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.decomposition_encdec import DecompositionDecoderLayer, DecompositionEncoderLayer
from tsflab.models._components.forecast_embedding import ForecastEmbedding
from tsflab.models._components.series_decomposition import SeriesDecomposition
from tsflab.models._inputs import check_window


def fft_autocorrelation(query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    """Equation (5), evaluated for every circular delay with an FFT."""
    if query.shape != key.shape or query.ndim != 4:
        raise ValueError("query and key must share (batch, heads, time, features)")
    length = query.shape[2]
    query_frequency = torch.fft.rfft(query, dim=2)
    key_frequency = torch.fft.rfft(key, dim=2)
    return torch.fft.irfft(
        query_frequency * torch.conj(key_frequency), n=length, dim=2
    )


class AutoCorrelation(nn.Module):
    """Paper equation (6): top-delay discovery followed by rolled aggregation."""

    def __init__(self, d_model: int, n_heads: int, factor: float, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.factor = factor
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.last_delays: torch.Tensor | None = None

    @staticmethod
    def _resize(values: torch.Tensor, length: int) -> torch.Tensor:
        if values.shape[1] == length:
            return values
        return F.interpolate(
            values.transpose(1, 2), size=length, mode="linear", align_corners=False
        ).transpose(1, 2)

    def forward(self, query: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        context = query if context is None else self._resize(context, query.shape[1])
        batch, length, width = query.shape
        q = self.query(query).view(batch, length, self.n_heads, self.head_dim)
        k = self.key(context).view(batch, length, self.n_heads, self.head_dim)
        v = self.value(context).view(batch, length, self.n_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        correlation = fft_autocorrelation(q, k).mean(dim=(1, 3))
        count = max(1, min(length, int(self.factor * math.log(max(length, 2)))))
        confidences, delays = torch.topk(correlation, count, dim=-1)
        weights = torch.softmax(confidences, dim=-1)
        aggregated = torch.zeros_like(v)
        for rank in range(count):
            shifted = torch.stack(
                [
                    torch.roll(v[item], shifts=-int(delays[item, rank]), dims=1)
                    for item in range(batch)
                ],
                dim=0,
            )
            aggregated = aggregated + shifted * weights[:, rank, None, None, None]
        self.last_delays = delays.detach()
        merged = aggregated.permute(0, 2, 1, 3).reshape(batch, length, width)
        return self.output(self.dropout(merged))


class AutoformerEncoderLayer(DecompositionEncoderLayer):
    """Equation (3): correlation and feed-forward, each followed by decomposition."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, moving_avg: int, factor: float, dropout: float, activation: str) -> None:
        super().__init__(
            AutoCorrelation(d_model, n_heads, factor, dropout),
            d_model, d_ff, moving_avg, dropout, activation, mixer_name="correlation",
        )


class AutoformerDecoderLayer(DecompositionDecoderLayer):
    """Equation (4): three decompositions and progressive trend accumulation."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, moving_avg: int, factor: float, dropout: float, activation: str, c_out: int) -> None:
        super().__init__(
            AutoCorrelation(d_model, n_heads, factor, dropout),
            AutoCorrelation(d_model, n_heads, factor, dropout),
            d_model, d_ff, moving_avg, dropout, activation, c_out,
            mixer_names=("self_correlation", "cross_correlation"),
        )


def _check_sizes(seq_len: int, label_len: int, pred_len: int) -> None:
    if label_len > seq_len:
        raise ValueError("label_len must not exceed seq_len")


class Model(nn.Module):
    """Forecast-only Autoformer with the paper's decomposition/correlation path."""

    def __init__(self, seq_len: int, label_len: int, pred_len: int, enc_in: int, dec_in: int, c_out: int, d_model: int, n_heads: int, e_layers: int, d_layers: int, d_ff: int, moving_avg: int, factor: float, dropout: float, activation: str = "gelu") -> None:
        super().__init__()
        _check_sizes(seq_len, label_len, pred_len)
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.decomposition = SeriesDecomposition(moving_avg)
        self.encoder_embedding = ForecastEmbedding(enc_in, d_model, dropout)
        self.decoder_embedding = ForecastEmbedding(dec_in, d_model, dropout)
        self.encoder = nn.ModuleList([AutoformerEncoderLayer(d_model, n_heads, d_ff, moving_avg, factor, dropout, activation) for _ in range(e_layers)])
        self.encoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.ModuleList([AutoformerDecoderLayer(d_model, n_heads, d_ff, moving_avg, factor, dropout, activation, c_out) for _ in range(d_layers)])
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
        if x_mark_enc.shape[:2] != x_enc.shape[:2]:
            raise ValueError("encoder marks do not align with x_enc")
        if x_mark_dec.shape[:2] != (x_enc.shape[0], decoder_length):
            raise ValueError("decoder marks do not align with the forecast decoder")
        return x_mark_enc, x_mark_dec

    def _decoder_init(self, x_enc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Seasonal (label part, then zeros) and trend (label part, then the input mean) decoder inputs."""
        seasonal_history, trend_history = self.decomposition(x_enc)
        mean_future = x_enc.mean(dim=1, keepdim=True).expand(-1, self.pred_len, -1)
        zero_future = torch.zeros_like(mean_future)
        if self.label_len:
            seasonal_init = torch.cat((seasonal_history[:, -self.label_len :], zero_future), dim=1)
            trend = torch.cat((trend_history[:, -self.label_len :], mean_future), dim=1)
        else:
            seasonal_init = zero_future
            trend = mean_future
        return seasonal_init, trend

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        self._check_inputs(x_enc, x_dec)
        x_mark_enc, x_mark_dec = self._marks(x_enc, x_mark_enc, x_mark_dec)
        seasonal_init, trend = self._decoder_init(x_enc)

        memory = self.encoder_embedding(x_enc, x_mark_enc)
        for layer in self.encoder:
            memory = layer(memory)
        memory = self.encoder_norm(memory)
        seasonal = self.decoder_embedding(seasonal_init, x_mark_dec)
        for layer in self.decoder:
            seasonal, trend = layer(seasonal, memory, trend)
        seasonal = self.seasonal_projection(self.decoder_norm(seasonal))
        return (seasonal + trend)[:, -self.pred_len :]
