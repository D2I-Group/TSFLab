"""Independent WaveNet-style dilated causal convolution forecaster."""

from __future__ import annotations

import torch
from torch import nn

from tsflab.models._components.gated_dilated_conv import gated_dilated_conv
from tsflab.models._components.revin import RevIN


class GatedCausalLayer(nn.Module):
    """Paper gated activation with residual and skip paths."""

    def __init__(
        self,
        residual_width: int,
        dilation_width: int,
        skip_width: int,
        kernel_size: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.filter = nn.Conv1d(
            residual_width, dilation_width, kernel_size, dilation=dilation
        )
        self.gate = nn.Conv1d(
            residual_width, dilation_width, kernel_size, dilation=dilation
        )
        self.residual = nn.Conv1d(dilation_width, residual_width, 1)
        self.skip = nn.Conv1d(dilation_width, skip_width, 1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gated = gated_dilated_conv(values, self.filter, self.gate)
        return values + self.residual(gated), self.skip(gated)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        label_len: int = 0,
        features: str = "M",
        residual_channels: int = 16,
        dilation_channels: int = 16,
        skip_channels: int = 64,
        end_channels: int = 128,
        kernel_size: int = 2,
        blocks: int = 2,
        layers: int = 2,
        use_norm: bool = True,
    ) -> None:
        super().__init__()
        del label_len
        if min(seq_len, pred_len) < 1:
            raise ValueError("lengths must be positive")
        if features not in {"M", "S", "MS"}:
            raise ValueError("features must be M, S, or MS")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.revin = RevIN(enc_in, affine=True) if use_norm else None
        self.input_projection = nn.Conv1d(1, residual_channels, 1)
        self.causal_layers = nn.ModuleList(
            GatedCausalLayer(
                residual_channels,
                dilation_channels,
                skip_channels,
                kernel_size,
                2**layer,
            )
            for _ in range(blocks)
            for layer in range(layers)
        )
        self.final_skip = nn.Conv1d(residual_channels, skip_channels, 1)
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, end_channels, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.horizon_projection = nn.Linear(end_channels, pred_len)

    def forward(
        self,
        x_enc,
        x_mark_enc=None,
        x_dec=None,
        x_mark_dec=None,
    ):
        del x_mark_enc, x_dec, x_mark_dec
        if x_enc.ndim != 3 or x_enc.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"x_enc must have shape [batch, {self.seq_len}, {self.enc_in}]"
            )
        normalized = self.revin(x_enc, "norm") if self.revin is not None else x_enc
        batch = normalized.shape[0]
        values = normalized.transpose(1, 2).reshape(
            batch * self.enc_in, 1, self.seq_len
        )
        residual = self.input_projection(values)
        skips: list[torch.Tensor] = []
        for layer in self.causal_layers:
            residual, skip = layer(residual)
            skips.append(skip)
        summary = self.head(
            torch.stack(skips).sum(0) + self.final_skip(residual)
        ).squeeze(-1)
        forecast = self.horizon_projection(summary).view(
            batch, self.enc_in, self.pred_len
        ).transpose(1, 2)
        if self.revin is not None:
            forecast = self.revin(forecast, "denorm")
        return forecast[..., -1:] if self.features == "MS" else forecast
