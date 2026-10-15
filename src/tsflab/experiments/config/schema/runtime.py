"""Experiment identity, output directory, device, and worker settings."""

from pydantic import BaseModel, ConfigDict, Field


class ExperimentRuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: str = "cuda"
    use_multi_gpu: bool = False
    device_ids: list[int] = Field(default_factory=lambda: [0])
    amp: bool = False
    num_workers: int = 4


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str
    random_seed: int
    work_dir: str = "./work_dirs"
    runtime: ExperimentRuntimeConfig = ExperimentRuntimeConfig()
