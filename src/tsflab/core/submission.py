"""SubmissionReport — the self-contained, uploadable submission bundle.

A SubmissionReport packages the machine-readable results (RunRecords) together
with the human/audit evidence (trajectory jsonl + rendered report) and a
content-addressed file manifest, so the whole thing can be opened as a PR
against the HF Submissions repo and reviewed by a human (v1) or re-checked by an
agent (v2).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .constants import SCHEMA_VERSION, Track
from .dataset_spec import DatasetSpec
from .run_record import RunRecord


class FileRef(BaseModel):
    """A bundled file with its content hash."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Path relative to the submission directory root.")
    sha256: str
    bytes: int | None = None
    role: str | None = Field(default=None, description="record | trajectory | report | asset")


class TrajectoryRef(BaseModel):
    """Reference to a captured (or synthesized) trajectory.jsonl."""

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    synthetic: bool = Field(default=False, description="True if reconstructed, not natively captured.")
    n_events: int | None = None


class ReportArtifact(BaseModel):
    """A rendered, human-readable report file."""

    model_config = ConfigDict(extra="forbid")

    path: str
    sha256: str
    format: str = Field(description="html | md")


class SubmissionManifest(BaseModel):
    """Manifest header: identity + content-addressed file list."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    submission_id: str
    submitter: str | None = None
    track: Track
    created_at: str | None = None
    files: list[FileRef] = Field(default_factory=list)
    files_sha256: str | None = Field(
        default=None,
        description="Self-hash over the canonicalized `files` array (bundle fingerprint).",
    )


class SubmissionReport(BaseModel):
    """Top-level submission bundle."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    manifest: SubmissionManifest
    datasets: list[DatasetSpec] = Field(default_factory=list)
    records: list[RunRecord] = Field(min_length=1)
    trajectories: list[TrajectoryRef] = Field(default_factory=list)
    reports: list[ReportArtifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistency(self) -> SubmissionReport:
        for record in self.records:
            if record.track != self.manifest.track:
                raise ValueError(
                    f"record {record.record_id!r} track {record.track!r} != manifest track {self.manifest.track!r}"
                )
        seen: set[str] = set()
        for ref in self.manifest.files:
            if ref.sha256 in seen:
                raise ValueError(f"duplicate file sha256 in manifest: {ref.sha256}")
            seen.add(ref.sha256)
        return self
