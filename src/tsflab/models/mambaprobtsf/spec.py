"""Runtime specification for MambaProbTSF."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mambaprobtsf.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=512, gt=0)
    d_state: int = Field(default=16, gt=0)
    d_ff: int = Field(default=512, gt=0)
    e_layers: int = Field(default=3, gt=0)
    d_conv: int = Field(default=2, gt=0)
    expand: int = Field(default=1, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"
    use_norm: bool = True
    sigma_network: Literal["mlp", "s_mamba"] = "mlp"
    sigma_hidden: int = Field(default=512, gt=0)
    use_marks: bool = True
    freq: Literal["h"] = "h"


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="MambaProbTSF",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_mambaprobtsf.toml",
    capabilities=frozenset({"distribution-output", "time-series"}),
    components=("embed", "mamba", "marks", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
