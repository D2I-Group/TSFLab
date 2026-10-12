"""Runtime specification for KARMA."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.karma.model import ATCD_HEADS, Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=512, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_state: int = Field(default=32, gt=0)
    d_conv: int = Field(default=4, gt=0)
    expand: int = Field(default=2, gt=0)
    embed_dim: int = Field(default=128, gt=0)
    decomposition: Literal["atcd", "moving_avg"] = "atcd"
    moving_avg: int = Field(default=25, gt=0)
    use_decomp: bool = True
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    use_norm: bool = True
    use_marks: bool = True
    freq: Literal["h"] = "h"
    time_loss_weight: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _contract(self) -> "ModelParameterConfig":
        if self.d_model % 2 or self.d_model < 8:
            raise ValueError("d_model must be even and at least 8 for the db4 transform")
        if self.embed_dim % ATCD_HEADS:
            raise ValueError(f"embed_dim must be divisible by {ATCD_HEADS} (ATCD attention heads)")
        if self.moving_avg % 2 == 0:
            raise ValueError("moving_avg must be odd")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


def training_objective(model, batch: TrainingBatch, criterion):
    """Eq. (2): the configured time-domain loss weighted by ``time_loss_weight``
    plus the complementary weight on the horizon-axis rfft MAE."""
    forecast = batch.forecast(model)
    aligned, target = batch.align(forecast), batch.target
    return forecast, model.hybrid_loss(aligned, target, criterion(aligned, target))


SPEC = ModelSpec(
    name="KARMA",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_karma.toml",
    capabilities=frozenset({"time-series"}),
    components=("embed", "mamba", "marks", "orthogonal_dwt", "revin", "series_decomposition"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
)
