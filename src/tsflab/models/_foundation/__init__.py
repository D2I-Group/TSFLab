"""Thin interfaces for official time-series foundation-model runtimes."""

from tsflab.models._foundation.official import (
    ChronosRuntime,
    MoiraiRuntime,
    TimesFMRuntime,
)
from tsflab.models._foundation.runtime import (
    FoundationDependencyError,
    FoundationForecast,
    FoundationModel,
    FoundationRuntime,
    FoundationSource,
)

__all__ = [
    "ChronosRuntime",
    "FoundationDependencyError",
    "FoundationForecast",
    "FoundationModel",
    "FoundationRuntime",
    "FoundationSource",
    "MoiraiRuntime",
    "TimesFMRuntime",
]
