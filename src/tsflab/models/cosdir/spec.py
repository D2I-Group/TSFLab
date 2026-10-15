"""Runtime specification for CosDir."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.cosdir.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    kernel_size: int = Field(default=25, gt=0)
    individual: bool = False
    weighting: Literal["fixed", "uw"] = "fixed"
    dir_lambda: float = Field(default=0.5, ge=0)
    eps: float = Field(default=1e-8, gt=0)

    @field_validator("kernel_size")
    @classmethod
    def _odd_kernel(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("kernel_size must be odd")
        return value


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        kernel_size=params["kernel_size"],
        individual=params["individual"],
        weighting=params["weighting"],
        dir_lambda=params["dir_lambda"],
        eps=params["eps"],
    )


def training_objective(model, batch, criterion):
    """Configured base loss plus the CosDir direction term on the target channels."""
    target_start = -1 if batch.features == "MS" else 0
    last_observed = batch.x[:, -1:, target_start:]
    forecast, loss, _ = model.training_objective(batch.x, batch.target, last_observed, criterion, batch.align)
    return forecast, loss


SPEC = ModelSpec(
    name="CosDir",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_cosdir.toml",
    capabilities=frozenset({"time-series"}),
    components=("dlinear",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
)
