"""Independent causal temporal-convolution forecasting baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from tsflab.models._components.revin import RevIN
from tsflab.models._inputs import check_window


class CausalConv1d(nn.Conv1d):
    """Left-pad a convolution so output at time t sees only inputs at or before t."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation)
        self.left_padding = (kernel_size - 1) * dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(F.pad(x, (self.left_padding, 0)))


class TemporalResidualBlock(nn.Module):
    """Two dilated causal convolutions followed by a residual addition."""

    def __init__(self, in_channels: int, out_channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, 3, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, 3, dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.activation(self.conv1(x)))
        hidden = self.dropout(self.activation(self.conv2(hidden)))
        return self.activation(hidden + self.residual(x))


class Model(nn.Module):
    """Encode with exponentially dilated residual blocks and a direct head."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        dropout: float = 0.1,
        num_layers: int = 2,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len, self.pred_len, self.enc_in = seq_len, pred_len, enc_in
        self.revin = RevIN(enc_in, enabled=use_revin)
        blocks: list[nn.Module] = []
        channels = enc_in
        for level in range(num_layers):
            blocks.append(TemporalResidualBlock(channels, d_model, 2**level, dropout))
            channels = d_model
        self.encoder = nn.Sequential(*blocks)
        self.head = nn.Linear(d_model, pred_len * enc_in)
        self.aux_loss: torch.Tensor | None = None

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        check_window(x_enc, self.seq_len, self.enc_in)
        normalized = self.revin(x_enc, "norm")
        encoded = self.encoder(normalized.transpose(1, 2))
        forecast = self.head(encoded[..., -1]).reshape(-1, self.pred_len, self.enc_in)
        forecast = self.revin(forecast, "denorm")
        self.aux_loss = forecast.new_zeros(())
        return forecast
