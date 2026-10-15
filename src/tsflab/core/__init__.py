"""TSF-Core — the contract layer shared by TSFLab (producer) and TSFLab Leaderboard (consumer).

This is a dependency-light pydantic package (pydantic + stdlib only). It defines
the contract models — :class:`DatasetSpec`, :class:`RunRecord`,
:class:`SubmissionReport`, and the real-time :class:`RoundSpec`,
:class:`ForecastSubmission`, and :class:`RoundScore` — and exports them to JSON Schema, which is the *only*
artifact TSFLab Leaderboard reads. Do not import ``benchmark`` / ``data`` / ``models`` /
``torch`` / ``numpy`` / ``datasets`` from here: that boundary is what keeps the
consumer side free of the ML stack.
"""

from __future__ import annotations

from .constants import (
    METRIC_NAMES,
    PROFILE_FIELDS,
    SCHEMA_VERSION,
    TaskMode,
    Track,
)
from .dataset_spec import DatasetSpec
from .export import export_schemas, iter_models
from .realtime import ForecastSubmission, RoundScore, RoundSpec
from .run_record import (
    HorizonResult,
    MetricSet,
    ProfileStats,
    RunConfigRef,
    RunEnv,
    RunRecord,
    RunTiming,
)
from .submission import (
    FileRef,
    ReportArtifact,
    SubmissionManifest,
    SubmissionReport,
    TrajectoryRef,
)

__all__ = [
    "SCHEMA_VERSION",
    "TaskMode",
    "Track",
    "METRIC_NAMES",
    "PROFILE_FIELDS",
    "DatasetSpec",
    "MetricSet",
    "ProfileStats",
    "RunTiming",
    "HorizonResult",
    "RunConfigRef",
    "RunEnv",
    "RunRecord",
    "FileRef",
    "TrajectoryRef",
    "ReportArtifact",
    "SubmissionManifest",
    "SubmissionReport",
    "ForecastSubmission",
    "RoundScore",
    "RoundSpec",
    "iter_models",
    "export_schemas",
]
