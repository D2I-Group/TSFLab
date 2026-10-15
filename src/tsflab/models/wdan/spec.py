"""Runtime specification for WDAN."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.wdan.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    wavelet: Literal["haar", "db4", "coif3"] = "coif3"
    dwt_levels: int = Field(default=1, gt=0)
    window_len: int = Field(default=5, gt=0)
    stat_d_model: int = Field(default=512, gt=0)
    stat_d_ff: int = Field(default=1024, gt=0)
    stat_dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    stat_ffn_layers: int = Field(default=2, ge=0)
    stat_pretrain_epochs: int = Field(default=5, ge=0)
    stat_lr: float = Field(default=1e-4, gt=0)
    joint_after_epochs: int = Field(default=1, ge=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    e_layers: int = Field(default=3, gt=0)
    d_model: int = Field(default=512, gt=0)
    n_heads: int = Field(default=8, gt=0)
    d_ff: int = Field(default=2048, gt=0)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _contract(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


def training_setup(model, train_loader, *, pred_len, features):
    """Fresh catalog training: record batches per epoch for the stage-2/3 switch."""
    del pred_len, features
    model.prepare_stages(len(train_loader))


def training_objective(model, batch, criterion):
    """Configured observation loss on the denormalized forecast (Eqs. 16-17); the
    statistics predictor is frozen until ``joint_after_epochs`` epochs have passed."""
    model.advance_stage()
    forecast = batch.forecast(model)
    return forecast, criterion(batch.align(forecast), batch.target)


SPEC = ModelSpec(
    name="WDAN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_wdan.toml",
    capabilities=frozenset({"time-series", "pretraining-stage"}),
    components=("patchtst",),
    contract_task={"seq_len": 96, "pred_len": 24, "label_len": 0},
    training_objective=training_objective,
    training_setup=training_setup,
)
