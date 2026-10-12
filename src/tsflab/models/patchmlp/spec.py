"""Model specification for PatchMLP."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.patchmlp.model import Model

from pydantic import BaseModel, Field, field_validator


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=1024, ge=4, multiple_of=4)
    e_layers: int = Field(default=1, gt=0)
    use_norm: bool = True
    moving_avg: int = Field(default=13, gt=0)
    patch_len: list[int] = Field(
        default_factory=lambda: [48, 24, 12, 6], min_length=4, max_length=4
    )

    @field_validator("moving_avg")
    @classmethod
    def _odd_moving_average(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("moving_avg must be odd")
        return value

    @field_validator("patch_len")
    @classmethod
    def _patch_lengths(cls, values: list[int]) -> list[int]:
        if any(value < 2 for value in values):
            raise ValueError("patch lengths must be at least two")
        return values


def build_model(cfg, params):
    """Construct PatchMLP from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        e_layers=params["e_layers"],
        use_norm=bool(params["use_norm"]),
        moving_avg=params["moving_avg"],
        patch_len=params.get("patch_len"),
    )


SPEC = ModelSpec(
    name="PatchMLP",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
