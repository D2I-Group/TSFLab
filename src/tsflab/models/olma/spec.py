"""Runtime specification for OLMA."""

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.olma.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    kernel_size: int = Field(default=25, gt=0)
    individual: bool = False
    loss_weight: float = Field(default=1.0, gt=0)
    weight_channel: float = Field(default=0.34, ge=0)
    weight_temporal: float = Field(default=0.33, ge=0)
    weight_wavelet: float = Field(default=0.33, ge=0)

    @field_validator("kernel_size")
    @classmethod
    def _odd_kernel(cls, value: int) -> int:
        if value % 2 == 0:
            raise ValueError("kernel_size must be odd")
        return value


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


def training_objective(model, batch: TrainingBatch, criterion):
    """Replace the configured time-domain criterion by the OLMA loss (Eq. 15)."""
    forecast = batch.forecast(model)
    loss, _ = model.olma_loss(batch.align(forecast), batch.target)
    return forecast, loss


SPEC = ModelSpec(
    name="OLMA",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_olma.toml",
    capabilities=frozenset({"time-series"}),
    components=("dlinear", "haar_dwt1d"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
)
