"""Runtime specification for DRFormer."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.drformer.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    d_ff: int = Field(default=256, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=2, gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=4, gt=0)
    sequence_num: int = Field(default=3, ge=1)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"
    mask_groups: int = Field(default=8, gt=0)
    active_ratio: float = Field(default=0.5, gt=0.0, le=1.0)
    death_rate: float = Field(default=0.5, ge=0.0, le=1.0)
    update_frequency: float = Field(default=0.3, gt=0.0)
    mask_epochs: float = Field(default=5.0, ge=0.0)

    @model_validator(mode="after")
    def _shape_contract(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even integer (rotary pairs)")
        if self.d_model % self.mask_groups or self.patch_len % self.mask_groups:
            raise ValueError("d_model and patch_len must be divisible by mask_groups")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


def training_setup(model, train_loader, *, pred_len, features):
    """Record the epoch length that drives the dynamic sparse indicator schedule."""
    del pred_len, features
    model.steps_per_epoch.fill_(len(train_loader))


def training_objective(model, batch, criterion):
    """Configured loss (MSE in the paper, Eq. 16) plus Algorithm 1's indicator updates."""
    model.advance_sparse_schedule()
    forecast = batch.forecast(model)
    loss = criterion(batch.align(forecast), batch.target)
    model.train_steps += 1
    return forecast, loss


SPEC = ModelSpec(
    name="DRFormer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_drformer.toml",
    capabilities=frozenset({"time-series"}),
    components=("flatten_forecast_head", "revin", "self_attention_family", "transformer_encdec"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
    training_setup=training_setup,
)
