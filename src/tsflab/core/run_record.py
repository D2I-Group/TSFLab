"""RunRecord — one (model x dataset x seed) result, horizon-folded.

The fixed ``MetricSet`` / ``ProfileStats`` schemas (10 optional slots each, keyed
byte-for-byte to the TSFLab producers) are what structurally eliminate the
``performance.csv`` column-drift bug: a missing metric is ``null``, never a
shifted column. Consumers always read by name.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import SCHEMA_VERSION, TaskMode, Track


class MetricSet(BaseModel):
    """The default metric suite. Keys match collect_metrics() exactly so a row
    can be built with ``MetricSet(**metrics_dict)`` and no key translation."""

    model_config = ConfigDict(extra="forbid")

    mae: float | None = None
    mse: float | None = None
    rmse: float | None = None
    mape: float | None = None
    mspe: float | None = None
    corr: float | None = None
    rse: float | None = None
    wape: float | None = None
    smape: float | None = None
    mase: float | None = None
    # Probabilistic metrics remain optional so point-forecast records validate.
    crps: float | None = None
    wql: float | None = None
    coverage_80: float | None = None
    width_80: float | None = None


class ProfileStats(BaseModel):
    """Architecture / performance profile. The two param counts are integers;
    the remaining fields are kept as the raw unit-bearing strings emitted by
    parse_profile_report() (e.g. '4.80 MB', '1.2345 ms')."""

    model_config = ConfigDict(extra="forbid")

    total_params: int | None = None
    trainable_params: int | None = None
    non_trainable_params: int | None = None
    total_mult_adds_mb: str | None = None
    total_macs_m: str | None = None
    dynamic_vram_mb: str | None = None
    peak_vram_mb: str | None = None
    reserved_vram_mb: str | None = None
    latency_avg_ms: str | None = None
    throughput_samples_sec: str | None = None


class RunTiming(BaseModel):
    """Wall-clock timing for a run (seconds)."""

    model_config = ConfigDict(extra="forbid")

    fit_time_sec: float | None = None
    inference_time_sec: float | None = None


class HorizonResult(BaseModel):
    """Metrics + profile for a single prediction horizon (= pred_len)."""

    model_config = ConfigDict(extra="forbid")

    horizon: int = Field(description="Prediction length (pred_len) this result is for.")
    metrics: MetricSet
    timing: RunTiming = Field(default_factory=RunTiming)
    profile: ProfileStats | None = None
    n_windows: int | None = Field(default=None, description="Number of evaluation windows.")
    run_id: str | None = Field(default=None, description="TSFLab run_id that produced this.")


class RunConfigRef(BaseModel):
    """Pinned config coordinates + optional resolved-config snapshot."""

    model_config = ConfigDict(extra="forbid")

    mode: TaskMode
    seq_len: int
    label_len: int
    pred_len: int
    features: str = "M"
    config_sha256: str | None = None
    snapshot: dict | None = Field(default=None, description="Resolved config model_dump (optional).")


class RunEnv(BaseModel):
    """Execution environment captured for reproducibility."""

    model_config = ConfigDict(extra="forbid")

    python: str | None = None
    torch: str | None = None
    cuda: str | None = None
    gpu: str | None = None
    platform: str | None = None
    framework_version: str | None = Field(default=None, description="TSFLab version.")
    git_sha: str | None = None
    git_dirty: bool | None = None


class RunRecord(BaseModel):
    """One result = one (model x dataset x seed), folded over horizons."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    record_id: str = Field(description="Stable id, e.g. '{model}__{dataset}__seed{seed}'.")
    model: str
    dataset_id: str
    track: Track
    mode: TaskMode
    seed: int
    seeds: list[int] = Field(default_factory=list, description="All seeds if aggregated.")
    results: list[HorizonResult] = Field(min_length=1)
    config: RunConfigRef
    env: RunEnv = Field(default_factory=RunEnv)
    created_at: str | None = Field(default=None, description="ISO-8601 UTC timestamp.")

    @model_validator(mode="after")
    def _consistency(self) -> RunRecord:
        horizons = [r.horizon for r in self.results]
        if len(set(horizons)) != len(horizons):
            raise ValueError("RunRecord.results has duplicate horizon entries")
        if self.config.mode != self.mode:
            raise ValueError("RunRecord.config.mode must equal RunRecord.mode")
        return self
