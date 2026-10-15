"""Rolling real-time evaluation contracts: rounds, forecast submissions, scores.

A real-time track advances in *rounds*. Each data release opens a round whose
``cutoff`` is the last observed timestamp; forecasts for the next ``horizon``
steps must be submitted before the first target timestamp is observed, and are
scored once a later release contains the complete target window. Because
forecasts are submitted before their truth exists, no participant (and no
model trained by the platform) can have seen the evaluation data.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import SCHEMA_VERSION, TaskMode


def _parse(ts: str) -> datetime:
    """Parse ISO 8601; naive values are interpreted as UTC."""
    value = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


class RoundSpec(BaseModel):
    """One evaluation round of a real-time track."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    track: str = Field(description="Real-time track id, e.g. 'traffic_pems_sb'.")
    round_id: str = Field(description="Round id, e.g. '2026-W40'.")
    data_version: str = Field(description="Dataset release the round was opened from.")
    hf_revision: str | None = Field(default=None, description="Pinned hub revision of that release.")
    mode: TaskMode = "time_series"
    freq: str = Field(description="Pandas frequency string of the panel, e.g. 'h' or 'B'.")
    cutoff: str = Field(description="Last observed timestamp (ISO 8601).")
    target_timestamps: list[str] = Field(description="Timestamps to forecast, in order.")
    channels: list[str] = Field(description="Channel (series / node) ids, in column order.")
    seq_len: int = Field(ge=1, description="Reference input length for catalog baselines.")
    opened_at: str = Field(description="When the round was opened (ISO 8601).")
    deadline: str = Field(description="Submissions after this instant are rejected.")
    norm_mean: list[float] = Field(description="Per-channel mean frozen at the cutoff.")
    norm_std: list[float] = Field(description="Per-channel std frozen at the cutoff.")

    @property
    def horizon(self) -> int:
        return len(self.target_timestamps)

    @model_validator(mode="after")
    def _consistent(self) -> RoundSpec:
        n = len(self.channels)
        if not self.target_timestamps:
            raise ValueError("a round needs at least one target timestamp")
        if len(self.norm_mean) != n or len(self.norm_std) != n:
            raise ValueError("normalization statistics must cover every channel")
        if _parse(self.deadline) > _parse(self.target_timestamps[0]):
            raise ValueError("deadline must not be later than the first target timestamp")
        return self


class ForecastSubmission(BaseModel):
    """Point forecasts for one round, submitted before the truth is observed."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    track: str
    round_id: str
    model: str = Field(description="Method name; catalog names bind to TSFLab specs.")
    submitter: str | None = None
    submitted_at: str = Field(description="Submission instant (ISO 8601), before the deadline.")
    predictions: list[list[float]] = Field(
        description="Forecast values, shape (horizon, channels), in the round's raw units."
    )
    code_revision: str | None = Field(default=None, description="Git commit that produced the forecast.")
    weights_uri: str | None = Field(default=None, description="Optional pinned hf:// weights bundle.")
    notes: str | None = None

    def check_against(self, spec: RoundSpec) -> None:
        """Raise ``ValueError`` unless the submission is valid for ``spec``."""
        if (self.track, self.round_id) != (spec.track, spec.round_id):
            raise ValueError("submission track/round does not match the round spec")
        if len(self.predictions) != spec.horizon:
            raise ValueError(f"expected {spec.horizon} forecast steps, got {len(self.predictions)}")
        width = len(spec.channels)
        for step, row in enumerate(self.predictions):
            if len(row) != width:
                raise ValueError(f"step {step}: expected {width} channels, got {len(row)}")
            if any(value != value or value in (float("inf"), float("-inf")) for value in row):
                raise ValueError(f"step {step}: forecasts must be finite")
        if _parse(self.submitted_at) > _parse(spec.deadline):
            raise ValueError("submission arrived after the round deadline")


class RoundScore(BaseModel):
    """Score of one submission once the round's truth is available."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    track: str
    round_id: str
    model: str
    submitter: str | None = None
    mse: float = Field(description="MSE on channels z-scored with the round's frozen statistics.")
    mae: float
    coverage: float = Field(ge=0, le=1, description="Fraction of target cells that were observed.")
    rank: int | None = None
