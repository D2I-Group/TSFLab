"""Runtime specification for SSCLForecaster."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.ssclforecaster.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=16, gt=0)
    d_ff: int = Field(default=16, gt=0)
    e_layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    scales: list[int] = Field(default_factory=lambda: [96], min_length=1)
    window_norm: Literal["last", "mean", "revin", "decomp"] = "last"
    autocon_lambda: float = Field(default=1.0, ge=0.0)
    use_marks: bool = True
    freq: Literal["h"] = "h"

    @field_validator("scales")
    @classmethod
    def _even_scales(cls, scales: list[int]) -> list[int]:
        if any(scale < 2 or scale % 2 for scale in scales):
            raise ValueError(
                "scales must be positive even lengths (moving-average kernels scale + 1)"
            )
        return scales


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


def training_setup(model: Model, train_loader, *, pred_len, features):
    """Appendix A.1: global autocorrelation of the smoothed training series."""
    del pred_len, features
    model.fit_autocorrelation(train_loader)


def training_objective(model: Model, batch: TrainingBatch, criterion):
    """Eq. (8): configured criterion + lambda * L_AutoCon."""
    forecast, representation = model.forecast_with_representation(batch.x, batch.x_mark)
    loss = criterion(batch.align(forecast), batch.target)
    return forecast, loss + model.autocon_lambda * model.autocon_loss(representation, batch.x_mark)


SPEC = ModelSpec(
    name="SSCLForecaster",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_ssclforecaster.toml",
    capabilities=frozenset({"time-series"}),
    components=("embed", "marks", "series_decomposition"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
    training_setup=training_setup,
)
