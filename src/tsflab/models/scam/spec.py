"""Runtime specification for SCAM."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.scam.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    candidates: int = Field(default=8, gt=0)
    hidden: int = Field(default=128, gt=1)
    snr: bool = False
    instance_norm: bool = True
    rec_layers: int = Field(default=4, gt=0)
    rec_width: int = Field(default=128, gt=0)
    rec_multiplier: int = Field(default=4, gt=0)
    loss_form: Literal["official", "paper"] = "official"

    @model_validator(mode="after")
    def _candidate_contract(self) -> ModelParameterConfig:
        if self.rec_width % self.candidates:
            raise ValueError("rec_width must be divisible by candidates")
        return self


def build_model(cfg, params):
    options = ModelParameterConfig(**params).model_dump()
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


def training_objective(model, batch, criterion):
    """Eq. (5): co-train the predictor ensemble and the relabeling network.

    Where a pseudo label ``g(y)`` lies outside the segment between ``y`` and the
    forecast, the supervised term is replaced by ``2 min(|g(y) - f(x)|, |g(y) - y|)``;
    the returned forecast is the ensemble mean.
    """
    del criterion
    channels = slice(-1, None) if batch.features == "MS" else slice(0, None)
    return model.training_loss(batch.x, batch.target, channels)


SPEC = ModelSpec(
    name="SCAM",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_scam.toml",
    capabilities=frozenset({"time-series"}),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
