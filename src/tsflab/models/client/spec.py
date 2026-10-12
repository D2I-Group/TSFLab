"""Runtime specification for Client."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.client.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    n_heads: int = Field(default=8, gt=0)
    e_layers: int = Field(default=2, gt=0)
    d_ff: int = Field(default=32, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"
    w_lin: float = 1.0


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        n_heads=params["n_heads"],
        e_layers=params["e_layers"],
        d_ff=params["d_ff"],
        dropout=params["dropout"],
        activation=params["activation"],
        w_lin=params["w_lin"],
    )


SPEC = ModelSpec(
    name="Client",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_client.toml",
    capabilities=frozenset({"time-series"}),
    components=("revin", "self_attention_family", "transformer_encdec"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
