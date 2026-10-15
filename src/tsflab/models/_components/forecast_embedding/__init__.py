"""Forecast value and raw-calendar embedding shared by decomposition models."""

from __future__ import annotations

import torch
from torch import nn


class RawCalendarEmbedding(nn.Module):
    """Project six raw calendar columns after fixed-range normalization."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(6, d_model, bias=False)

    def forward(self, marks: torch.Tensor) -> torch.Tensor:
        if marks.ndim != 3 or marks.shape[-1] != 6:
            raise ValueError("calendar marks must have shape (batch, time, 6)")
        scales = marks.new_tensor((2100.0, 12.0, 31.0, 6.0, 23.0, 59.0))
        return self.projection(marks / scales - 0.5)


class ForecastEmbedding(nn.Module):
    """Add projected values and normalized raw-calendar covariates."""

    def __init__(self, channels: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.value = nn.Linear(channels, d_model)
        self.calendar = RawCalendarEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError("values must have shape (batch, time, channels)")
        if marks.shape[:2] != values.shape[:2]:
            raise ValueError("values and calendar marks must share batch and time axes")
        return self.dropout(self.value(values) + self.calendar(marks))
