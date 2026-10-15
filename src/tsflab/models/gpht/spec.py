"""Runtime specification for GPHT."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gpht.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    token_len: int = Field(default=48, gt=0)
    pooling_rates: list[int] = Field(default_factory=lambda: [8, 4, 2, 1], min_length=1)
    d_model: int = Field(default=512, gt=0)
    d_ff: int = Field(default=2048, gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=3, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"

    @model_validator(mode="after")
    def _architecture_contract(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if any(rate < 1 or self.token_len % rate for rate in self.pooling_rates):
            raise ValueError("every pooling rate must divide token_len")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        token_len=params["token_len"],
        pooling_rates=tuple(params["pooling_rates"]),
        d_model=params["d_model"],
        d_ff=params["d_ff"],
        n_heads=params["n_heads"],
        e_layers=params["e_layers"],
        dropout=params["dropout"],
        activation=params["activation"],
    )


def training_objective(model, batch, criterion):
    """Auto-regressive next-token MSE over every position; the criterion is not used."""
    return None, model.training_objective(batch.x, batch.target)


SPEC = ModelSpec(
    name="GPHT",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_gpht.toml",
    capabilities=frozenset({"time-series"}),
    components=("embed", "revin", "self_attention_family", "transformer_encdec"),
    contract_task={"seq_len": 96, "pred_len": 48, "label_len": 0},
    training_objective=training_objective,
)
