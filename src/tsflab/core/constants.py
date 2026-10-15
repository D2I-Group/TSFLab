"""Shared constants for the TSF-Core contract layer.

This module is the single source of truth for the metric / profile field names
and the task-mode / track vocabularies used across DatasetSpec, RunRecord and
SubmissionReport. The metric and profile tuples are kept byte-for-byte aligned
with their TSFLab producers so that records can be assembled with zero
key translation:

- ``METRIC_NAMES``  ↔ ``tsflab.experiments.evaluation.metrics.collect_metrics`` keys
- ``PROFILE_FIELDS`` ↔ ``tsflab.experiments.evaluation.profile.parse_profile_report`` keys

TSF-Core depends only on ``pydantic`` + the Python standard library. It must
never import ``benchmark`` / ``data`` / ``models`` / ``torch`` / ``numpy`` /
``datasets`` — that boundary is what lets TSFLab Leaderboard consume the exported JSON
Schema without pulling the ML stack.
"""

from __future__ import annotations

from typing import Literal

SCHEMA_VERSION = "1.1.0"

TaskMode = Literal["time_series", "spatiotemporal", "covariate"]
Track = Literal["time_series", "spatiotemporal", "covariate", "realtime"]

# Default metric suite — must match collect_metrics() in
# src/tsflab/experiments/evaluation/metrics.py exactly (order included).
METRIC_NAMES: tuple[str, ...] = (
    "mae",
    "mse",
    "rmse",
    "mape",
    "mspe",
    "corr",
    "rse",
    "wape",
    "smape",
    "mase",
    "crps",
    "wql",
    "coverage_80",
    "width_80",
)

# Profile fields — must match parse_profile_report()'s prefix_map targets in
# src/tsflab/experiments/evaluation/profile.py. The two ``*_params`` are integers; the
# rest are kept as unit-bearing strings (faithful to the raw report) and are
# only structured into {value, unit} downstream if/when TSFLab Leaderboard needs to sort on
# them.
PROFILE_FIELDS: tuple[str, ...] = (
    "total_params",
    "trainable_params",
    "non_trainable_params",
    "total_mult_adds_mb",
    "total_macs_m",
    "dynamic_vram_mb",
    "peak_vram_mb",
    "reserved_vram_mb",
    "latency_avg_ms",
    "throughput_samples_sec",
)
