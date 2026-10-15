"""Runtime specification for ASGMamba."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.asgmamba.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    patch_sizes: list[int] = Field(default_factory=lambda: [8, 16, 32], min_length=1)
    overlap: bool = False
    d_model: int = Field(default=256, gt=0)
    d_state: int = Field(default=16, gt=0)
    d_conv: int = Field(default=4, gt=0)
    expand: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.2, ge=0.0, lt=1.0)
    gate: Literal["identity_centered", "sigmoid"] = "identity_centered"
    gate_amplitude: float = Field(default=0.5, gt=0.0, le=1.0)
    revin_affine: bool = True

    @field_validator("patch_sizes")
    @classmethod
    def _positive_patches(cls, value: list[int]) -> list[int]:
        if min(value) < 1:
            raise ValueError("patch_sizes must be positive")
        return value


def build_model(cfg, params):
    options = ModelParameterConfig(**params).model_dump()
    options["patch_sizes"] = tuple(options["patch_sizes"])
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


SPEC = ModelSpec(
    name="ASGMamba",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_asgmamba.toml",
    capabilities=frozenset({"time-series"}),
    components=("mamba", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
