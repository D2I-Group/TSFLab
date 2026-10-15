"""Runtime specification for Basisformer."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.model_io import slice_target
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.basisformer.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=100, gt=0)
    heads: int = Field(default=16, gt=0)
    basis_nums: int = Field(default=10, gt=0)
    block_nums: int = Field(default=2, ge=0)
    bottleneck: int = Field(default=2, gt=0)
    map_bottleneck: int = Field(default=20, gt=0)
    tau: float = Field(default=0.07, gt=0.0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    timestamp_origin_year: int = 2010
    timestamp_span_years: float = Field(default=10.0, gt=0.0)
    loss_weight_infonce: float = Field(default=1.0, ge=0.0)
    loss_weight_smooth: float = Field(default=1.0, ge=0.0)

    @model_validator(mode="after")
    def _architecture_contract(self) -> ModelParameterConfig:
        if self.d_model < self.heads:
            raise ValueError("d_model must be at least heads")
        return self


def build_model(cfg, params):
    options = ModelParameterConfig(**params).model_dump()
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


def training_objective(model: Model, batch: TrainingBatch, criterion):
    """Eq. (9): configured criterion + w_infonce * L_align + w_smooth * L_smooth."""
    future = slice_target(batch.y, batch.pred_len, "M")  # every channel forms c_y
    forecast, align, smooth = model.training_terms(batch.x, batch.x_mark, future)
    loss = criterion(batch.align(forecast), batch.target)
    loss = loss + model.loss_weight_infonce * align + model.loss_weight_smooth * smooth
    return forecast, loss


SPEC = ModelSpec(
    name="Basisformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_basisformer.toml",
    capabilities=frozenset({"time-series"}),
    components=("marks",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
