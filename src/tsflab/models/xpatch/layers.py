"""Paper-derived, model-local building blocks for the xPatch rewrite."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExponentialDecomposition(nn.Module):
    """Split a BLC sequence into residual seasonality and an EMA trend.

    EMA follows equation (2) of the paper. ``dema`` is an explicit local
    extension using Holt's level-and-trend double exponential recurrence.
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.3, kind: str = "ema") -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kind = kind

    @staticmethod
    def _ema(x: torch.Tensor, smoothing: float) -> torch.Tensor:
        states = [x[:, :1, :]]
        for step in range(1, x.shape[1]):
            states.append(smoothing * x[:, step : step + 1, :] + (1.0 - smoothing) * states[-1])
        return torch.cat(states, dim=1)

    def _dema(self, x: torch.Tensor) -> torch.Tensor:
        level = x[:, :1, :]
        slope = x[:, 1:2, :] - level if x.shape[1] > 1 else torch.zeros_like(level)
        states = [level]
        for step in range(1, x.shape[1]):
            previous = level
            level = self.alpha * x[:, step : step + 1, :] + (1.0 - self.alpha) * (level + slope)
            slope = self.beta * (level - previous) + (1.0 - self.beta) * slope
            states.append(level)
        return torch.cat(states, dim=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self._ema(x, self.alpha) if self.kind == "ema" else self._dema(x)
        return x - trend, trend


class LinearTrendStream(nn.Module):
    """Activation-free bottleneck with linear, pooling, and normalization blocks."""

    def __init__(self, seq_len: int, pred_len: int, hidden_dim: int) -> None:
        super().__init__()
        first_width = max(4, 2 * hidden_dim)
        second_width = max(4, hidden_dim)
        self.input_projection = nn.Linear(seq_len, first_width)
        self.first_norm = nn.LayerNorm(first_width // 2)
        self.bottleneck_projection = nn.Linear(first_width // 2, second_width)
        self.second_norm = nn.LayerNorm(second_width // 2)
        self.output_projection = nn.Linear(second_width // 2, pred_len)

    @staticmethod
    def _pool(x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool1d(x.unsqueeze(1), kernel_size=2, stride=2).squeeze(1)

    def forward(self, trend: torch.Tensor) -> torch.Tensor:
        hidden = self.first_norm(self._pool(self.input_projection(trend)))
        hidden = self.second_norm(self._pool(self.bottleneck_projection(hidden)))
        return self.output_projection(hidden)


class SafeBatchNorm1d(nn.BatchNorm1d):
    """Use stored statistics for the otherwise invalid single-value train case."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values_per_channel = x.numel() // x.shape[1]
        if self.training and values_per_channel == 1:
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                False,
                0.0,
                self.eps,
            )
        return super().forward(x)


class NonlinearPatchStream(nn.Module):
    """Channel-independent patch CNN matching equations (5)--(11)."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        padding_patch: str,
    ) -> None:
        super().__init__()
        if padding_patch == "none" and seq_len < patch_len:
            raise ValueError("patch_len cannot exceed seq_len without end padding")
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        padded_length = seq_len + self._padding_for(seq_len)
        self.num_patches = 1 + (padded_length - patch_len) // stride

        embedded_length = patch_len * patch_len
        self.patch_embedding = nn.Linear(patch_len, embedded_length)
        self.embedding_norm = SafeBatchNorm1d(self.num_patches)
        self.depthwise = nn.Conv1d(
            self.num_patches,
            self.num_patches,
            kernel_size=patch_len,
            stride=patch_len,
            groups=self.num_patches,
        )
        self.depthwise_norm = SafeBatchNorm1d(self.num_patches)
        self.pointwise = nn.Conv1d(self.num_patches, self.num_patches, kernel_size=1)
        self.pointwise_norm = SafeBatchNorm1d(self.num_patches)
        flattened = self.num_patches * patch_len
        self.forecast_head = nn.Sequential(
            nn.Linear(flattened, 2 * pred_len),
            nn.GELU(),
            nn.Linear(2 * pred_len, pred_len),
        )

    def _padding_for(self, length: int) -> int:
        if self.padding_patch == "none":
            return 0
        if length < self.patch_len:
            return self.patch_len - length
        return self.stride

    def extract_patches(self, seasonal: torch.Tensor) -> torch.Tensor:
        padding = self._padding_for(seasonal.shape[-1])
        if padding:
            seasonal = F.pad(seasonal, (0, padding), mode="replicate")
        return seasonal.unfold(-1, self.patch_len, self.stride)

    def forward(self, seasonal: torch.Tensor) -> torch.Tensor:
        patches = self.extract_patches(seasonal)
        embedded = self.embedding_norm(F.gelu(self.patch_embedding(patches)))
        residual = F.avg_pool1d(embedded, kernel_size=self.patch_len, stride=self.patch_len)
        hidden = self.depthwise_norm(F.gelu(self.depthwise(embedded))) + residual
        hidden = self.pointwise_norm(F.gelu(self.pointwise(hidden)))
        return self.forecast_head(hidden.flatten(1))


class DualStreamForecaster(nn.Module):
    """Forecast one normalized univariate series with linear and CNN flows."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        padding_patch: str,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.linear_stream = LinearTrendStream(seq_len, pred_len, hidden_dim)
        self.nonlinear_stream = NonlinearPatchStream(seq_len, pred_len, patch_len, stride, padding_patch)
        self.fusion = nn.Sequential(
            nn.Linear(2 * pred_len, 2 * pred_len),
            nn.GELU(),
            SafeBatchNorm1d(2 * pred_len),
            nn.Linear(2 * pred_len, pred_len),
        )

    def forward(self, seasonal: torch.Tensor, trend: torch.Tensor) -> torch.Tensor:
        linear = self.linear_stream(trend)
        nonlinear = self.nonlinear_stream(seasonal)
        return self.fusion(torch.cat((linear, nonlinear), dim=-1))
