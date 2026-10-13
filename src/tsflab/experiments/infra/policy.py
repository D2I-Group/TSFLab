"""Optional execution settings, separate from the scientific run configuration."""

from pathlib import Path
from typing import Literal
import tomllib

from pydantic import BaseModel, ConfigDict, Field


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Budget(StrictConfig):
    max_runs: int | None = Field(default=None, ge=1)
    max_parallel_jobs: int = Field(default=1, ge=1)
    auto_parallel: bool = False
    max_wall_minutes: float | None = Field(default=None, gt=0)
    run_timeout_minutes: float | None = Field(default=None, gt=0)
    max_gpu_hours: float | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, gt=0)
    max_retries: int = Field(default=0, ge=0, le=10)


class Resources(StrictConfig):
    auto_select: bool = False
    gpus: list[str] = Field(default_factory=list)
    sharing: bool = False
    max_processes_per_gpu: int = Field(default=2, ge=1)
    memory_per_run_mb: int = Field(default=0, ge=0)
    gpus_per_run: int = Field(default=1, ge=1)
    min_free_memory_mb: int = Field(default=0, ge=0)
    wait_timeout_minutes: float = Field(default=30, gt=0)
    min_free_disk_gb: float = Field(default=0.1, ge=0)
    # Caps OpenMP/MKL/OpenBLAS threads in each run process; None keeps library defaults.
    threads_per_run: int | None = Field(default=None, ge=1)


class Recovery(StrictConfig):
    checkpoint_every_batches: int = Field(default=0, ge=0)


class Storage(StrictConfig):
    max_run_gb: float | None = Field(default=None, gt=0)
    keep_epoch_checkpoints: int = Field(default=3, ge=0)


class Tracking(StrictConfig):
    prediction_samples: int = Field(default=0, ge=0, le=8)
    tensorboard: bool = False
    wandb: Literal["disabled", "offline", "online"] = "disabled"
    project: str = "TSFLab"
    entity: str | None = None
    tags: list[str] = Field(default_factory=list)


class ExecutionPolicy(StrictConfig):
    budget: Budget = Field(default_factory=Budget)
    resources: Resources = Field(default_factory=Resources)
    recovery: Recovery = Field(default_factory=Recovery)
    storage: Storage = Field(default_factory=Storage)
    tracking: Tracking = Field(default_factory=Tracking)


def load_policy(path: str | None) -> ExecutionPolicy:
    """Load optional TOML without changing the experiment's scientific identity."""
    if path is None:
        return ExecutionPolicy()
    with Path(path).open("rb") as stream:
        return ExecutionPolicy.model_validate(tomllib.load(stream))
