"""Runtime specification for OCETS."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.ocets.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    num_bins: int = Field(default=45, ge=2)
    sigma: float = Field(default=0.015, gt=0)
    kernel_size: int = Field(default=25, gt=0)
    individual: bool = False
    value_range: Literal["signed", "unit"] = "signed"

    @model_validator(mode="after")
    def _odd_kernel(self) -> "ModelParameterConfig":
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


def training_objective(model, batch, criterion):
    """Ordinal cross-entropy against truncated-Gaussian soft labels (replaces the configured loss)."""
    del criterion
    return model.training_objective(batch.x, batch.target)


SPEC = ModelSpec(
    name="OCETS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_ocets.toml",
    capabilities=frozenset({"time-series"}),
    components=("dlinear",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
)
