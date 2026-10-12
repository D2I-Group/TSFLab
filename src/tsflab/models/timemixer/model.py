"""Clean-room TimeMixer with paper-defined PDM and FMM paths."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._components.series_decomposition import EdgePaddedMovingAverage


class MovingAverageDecomposition(nn.Module):
    """Residual/trend split via the shared edge-padded moving average.

    Equivalent to replicate-padding the series by ``(kernel_size - 1) // 2``
    on both sides and average-pooling with stride 1; identical to
    ``tsflab.models._components.series_decomposition.EdgePaddedMovingAverage``.
    """

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self._moving_average = EdgePaddedMovingAverage(kernel_size, stride=1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self._moving_average(values)
        return values - trend, trend


class DFTDecomposition(nn.Module):
    """Keep the strongest non-DC frequencies as the seasonal component."""

    def __init__(self, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        spectrum = torch.fft.rfft(values, dim=1)
        amplitudes = spectrum.abs().clone()
        amplitudes[:, 0] = 0
        indices = torch.topk(amplitudes, self.top_k, dim=1).indices
        mask = torch.zeros_like(amplitudes, dtype=torch.bool)
        mask.scatter_(1, indices, True)
        seasonal_spectrum = torch.where(mask, spectrum, torch.zeros_like(spectrum))
        seasonal = torch.fft.irfft(seasonal_spectrum, n=values.shape[1], dim=1)
        return seasonal, values - seasonal


class TemporalMixer(nn.Module):
    """Two-layer GELU MLP operating only on the temporal dimension."""

    def __init__(self, input_length: int, output_length: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_length, output_length),
            nn.GELU(),
            nn.Linear(output_length, output_length),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class PastDecomposableMixing(nn.Module):
    """Paper equations (3)-(5): seasonal bottom-up and trend top-down mixing."""

    def __init__(self, lengths: tuple[int, ...], d_model: int, d_ff: int, moving_avg: int, top_k: int, decomp_method: str, dropout: float) -> None:
        super().__init__()
        if decomp_method == "moving_avg":
            self.decomposition: nn.Module = MovingAverageDecomposition(moving_avg)
        else:
            self.decomposition = DFTDecomposition(top_k)
        self.seasonal_bottom_up = nn.ModuleList(
            [TemporalMixer(lengths[index], lengths[index + 1]) for index in range(len(lengths) - 1)]
        )
        self.trend_top_down = nn.ModuleList(
            [TemporalMixer(lengths[index + 1], lengths[index]) for index in range(len(lengths) - 1)]
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, scales: list[torch.Tensor]) -> list[torch.Tensor]:
        decomposed = [self.decomposition(scale) for scale in scales]
        seasonal, trend = map(list, zip(*decomposed))
        for index, mixer in enumerate(self.seasonal_bottom_up):
            contribution = mixer(seasonal[index].transpose(1, 2)).transpose(1, 2)
            seasonal[index + 1] = seasonal[index + 1] + contribution
        for index in reversed(range(len(self.trend_top_down))):
            contribution = self.trend_top_down[index](
                trend[index + 1].transpose(1, 2)
            ).transpose(1, 2)
            trend[index] = trend[index] + contribution
        return [
            original + self.feed_forward(season + trend_component)
            for original, season, trend_component in zip(scales, seasonal, trend)
        ]


def multiscale_lengths(seq_len: int, window: int, layers: int) -> tuple[int, ...]:
    lengths = [seq_len]
    for _ in range(layers):
        next_length = lengths[-1] // window
        if next_length < 1:
            raise ValueError("down-sampling creates an empty scale")
        lengths.append(next_length)
    return tuple(lengths)


def _check_arguments(seq_len: int, pred_len: int) -> None:
    if min(seq_len, pred_len) < 1:
        raise ValueError("lengths, channels, widths, and layers must be positive")


def _check_top_k(decomp_method: str, top_k: int, scale_lengths: tuple[int, ...]) -> None:
    if decomp_method == "dft_decomp" and top_k > min(scale_lengths) // 2:
        raise ValueError("top_k exceeds the coarsest scale's non-DC frequencies")


class Model(nn.Module):
    """Forecast-only TimeMixer using average-pooled scales and sum-ensemble FMM."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, c_out: int, e_layers: int, d_model: int, d_ff: int, down_sampling_window: int, down_sampling_layers: int, moving_avg: int, top_k: int, dropout: float, use_norm: bool, decomp_method: str) -> None:
        super().__init__()
        _check_arguments(seq_len, pred_len)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.down_sampling_window = down_sampling_window
        self.scale_lengths = multiscale_lengths(
            seq_len, down_sampling_window, down_sampling_layers
        )
        _check_top_k(decomp_method, top_k, self.scale_lengths)
        self.normalizers = nn.ModuleList(
            [RevIN(enc_in, affine=True, enabled=use_norm) for _ in self.scale_lengths]
        )
        self.embeddings = nn.ModuleList(
            [nn.Linear(enc_in, d_model) for _ in self.scale_lengths]
        )
        self.pdm_blocks = nn.ModuleList(
            [
                PastDecomposableMixing(
                    self.scale_lengths,
                    d_model,
                    d_ff,
                    moving_avg,
                    top_k,
                    decomp_method,
                    dropout,
                )
                for _ in range(e_layers)
            ]
        )
        self.temporal_predictors = nn.ModuleList(
            [nn.Linear(length, pred_len) for length in self.scale_lengths]
        )
        self.channel_predictors = nn.ModuleList(
            [nn.Linear(d_model, c_out) for _ in self.scale_lengths]
        )

    def _downsample(self, values: torch.Tensor) -> list[torch.Tensor]:
        scales = [values]
        for _ in range(len(self.scale_lengths) - 1):
            values = F.avg_pool1d(
                values.transpose(1, 2),
                kernel_size=self.down_sampling_window,
                stride=self.down_sampling_window,
            ).transpose(1, 2)
            scales.append(values)
        return scales

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.shape[1:] != (self.seq_len, self.channels):
            raise ValueError(f"x_enc must have shape (batch, {self.seq_len}, {self.channels})")
        raw_scales = self._downsample(x_enc)
        normalized = [
            normalizer(scale, "norm")
            for normalizer, scale in zip(self.normalizers, raw_scales)
        ]
        encoded = [
            embedding(scale) for embedding, scale in zip(self.embeddings, normalized)
        ]
        for block in self.pdm_blocks:
            encoded = block(encoded)
        predictions = []
        for representation, temporal, channel in zip(
            encoded, self.temporal_predictors, self.channel_predictors
        ):
            future_features = temporal(representation.transpose(1, 2)).transpose(1, 2)
            predictions.append(channel(future_features))
        forecast = torch.stack(predictions, dim=-1).sum(dim=-1)
        return self.normalizers[0](forecast, "denorm")
