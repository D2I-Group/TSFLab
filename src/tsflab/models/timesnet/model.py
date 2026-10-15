"""Clean-room TimesNet forecast implementation from the paper's 2D equations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.dominant_periods import dominant_periods
from tsflab.models._components.forecast_embedding import RawCalendarEmbedding
from tsflab.models._components.inception_block import InceptionBlock2d
from tsflab.models._components.instance_norm import destandardize, standardize
from tsflab.models._inputs import check_marks, check_window


class TimesBlock(nn.Module):
    """Transform 1D variations into period-aligned 2D variations and aggregate."""

    def __init__(self, total_length: int, top_k: int, d_model: int, d_ff: int, num_kernels: int) -> None:
        super().__init__()
        self.total_length = total_length
        self.top_k = top_k
        self.convolution = nn.Sequential(
            InceptionBlock2d(d_model, d_ff, num_kernels, init_weight=False),
            nn.GELU(),
            InceptionBlock2d(d_ff, d_model, num_kernels, init_weight=False),
        )
        self.last_periods: torch.Tensor | None = None

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[1] != self.total_length:
            raise ValueError("TimesBlock received an unexpected temporal length")
        batch, length, channels = values.shape
        periods, amplitudes = dominant_periods(values, self.top_k)
        transformed = []
        for period_tensor in periods:
            period = max(1, int(period_tensor))
            padded_length = ((length + period - 1) // period) * period
            padded = F.pad(values, (0, 0, 0, padded_length - length))
            image = padded.reshape(batch, padded_length // period, period, channels)
            image = image.permute(0, 3, 1, 2).contiguous()
            encoded = self.convolution(image)
            transformed.append(encoded.permute(0, 2, 3, 1).reshape(batch, padded_length, channels)[:, :length])
        stacked = torch.stack(transformed, dim=-1)
        weights = torch.softmax(amplitudes, dim=-1)[:, None, None, :]
        self.last_periods = torch.as_tensor(periods, device=values.device)
        return values + (stacked * weights).sum(dim=-1)


class Model(nn.Module):
    """Forecast-only TimesNet with stacked residual TimesBlocks."""

    def __init__(
        self,
        seq_len: int,
        label_len: int,
        pred_len: int,
        enc_in: int,
        c_out: int,
        d_model: int,
        e_layers: int,
        d_ff: int,
        dropout: float,
        top_k: int = 3,
        num_kernels: int = 3,
    ) -> None:
        super().__init__()
        del label_len
        total_length = seq_len + pred_len
        if top_k > total_length // 2:
            raise ValueError("top_k exceeds the available non-DC Fourier frequencies")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.value_embedding = nn.Linear(enc_in, d_model)
        self.calendar_embedding = RawCalendarEmbedding(d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.temporal_projection = nn.Linear(seq_len, total_length)
        self.blocks = nn.ModuleList(
            [TimesBlock(total_length, top_k, d_model, d_ff, num_kernels) for _ in range(e_layers)]
        )
        self.block_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(e_layers)])
        self.output_projection = nn.Linear(d_model, c_out)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_dec, x_mark_dec
        check_window(x_enc, self.seq_len, self.channels)
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros(x_enc.shape[0], self.seq_len, 6)
        check_marks(x_mark_enc, x_enc)

        normalized, mean, stdev = standardize(x_enc)
        embedded = self.value_embedding(normalized) + self.calendar_embedding(x_mark_enc)
        embedded = self.embedding_dropout(embedded)
        encoded = self.temporal_projection(embedded.transpose(1, 2)).transpose(1, 2)
        for block, norm in zip(self.blocks, self.block_norms):
            encoded = norm(block(encoded))
        forecast = self.output_projection(encoded)
        forecast = destandardize(forecast, mean, stdev)
        return forecast[:, -self.pred_len :]
