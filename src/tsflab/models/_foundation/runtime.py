"""Canonical boundary between TSFLab and official pretrained runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch
from torch import nn


class FoundationDependencyError(RuntimeError):
    """An optional official runtime is unavailable or incompatible."""


@dataclass(frozen=True)
class FoundationSource:
    """Facts needed to reproduce one official runtime selection."""

    name: str
    package: str
    codebase: str
    model_id: str
    revision: str
    license: str

    def __post_init__(self) -> None:
        values = (
            self.name,
            self.package,
            self.codebase,
            self.model_id,
            self.revision,
            self.license,
        )
        if any(not value.strip() for value in values):
            raise ValueError("foundation source facts must be non-empty")
        if not self.codebase.startswith("https://"):
            raise ValueError("foundation source codebase must use https://")


@dataclass(frozen=True)
class FoundationForecast:
    """Normalized official forecast for flattened univariate series."""

    mean: torch.Tensor
    quantiles: torch.Tensor | None = None


@runtime_checkable
class FoundationRuntime(Protocol):
    """Small provider-neutral forecasting interface."""

    source: FoundationSource

    def predict(
        self,
        context: torch.Tensor,
        prediction_length: int,
        quantile_levels: tuple[float, ...],
    ) -> FoundationForecast:
        """Forecast a two-dimensional ``[series, time]`` context batch."""


def _validate_forecast(
    forecast: FoundationForecast,
    series: int,
    prediction_length: int,
    quantiles: int,
) -> None:
    expected_mean = (series, prediction_length)
    if tuple(forecast.mean.shape) != expected_mean:
        raise ValueError(f"foundation mean shape {tuple(forecast.mean.shape)} != {expected_mean}")
    if forecast.quantiles is not None:
        expected_quantiles = (series, prediction_length, quantiles)
        if tuple(forecast.quantiles.shape) != expected_quantiles:
            raise ValueError(f"foundation quantile shape {tuple(forecast.quantiles.shape)} != {expected_quantiles}")
    if not torch.isfinite(forecast.mean).all():
        raise ValueError("foundation runtime returned non-finite mean values")
    if forecast.quantiles is not None and not torch.isfinite(forecast.quantiles).all():
        raise ValueError("foundation runtime returned non-finite quantile values")


class FoundationModel(nn.Module):
    """Expose an official runtime through the canonical four-input model API."""

    def __init__(
        self,
        runtime: FoundationRuntime,
        prediction_length: int,
        quantile_levels: tuple[float, ...] = (),
    ) -> None:
        super().__init__()
        if prediction_length < 1:
            raise ValueError("prediction_length must be positive")
        if quantile_levels and any(left >= right for left, right in zip(quantile_levels, quantile_levels[1:])):
            raise ValueError("quantile_levels must be strictly increasing")
        if any(not 0 < level < 1 for level in quantile_levels):
            raise ValueError("quantile_levels must lie strictly between zero and one")
        self.runtime = runtime
        self.prediction_length = prediction_length
        self.quantile_levels = tuple(float(level) for level in quantile_levels)
        self.output_type = "quantile" if self.quantile_levels else "point"

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None):
        if x_enc.ndim != 3:
            raise ValueError("foundation input must have shape [batch, time, channels]")
        batch, _, channels = x_enc.shape
        context = x_enc.permute(0, 2, 1).reshape(batch * channels, -1)
        forecast = self.runtime.predict(context, self.prediction_length, self.quantile_levels)
        _validate_forecast(
            forecast,
            series=batch * channels,
            prediction_length=self.prediction_length,
            quantiles=len(self.quantile_levels),
        )
        if self.quantile_levels:
            if forecast.quantiles is None:
                raise ValueError("quantile runtime did not return quantiles")
            return forecast.quantiles.reshape(
                batch, channels, self.prediction_length, len(self.quantile_levels)
            ).permute(0, 2, 1, 3)
        return forecast.mean.reshape(batch, channels, self.prediction_length).permute(0, 2, 1)
