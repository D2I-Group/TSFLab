"""Runtime specification for SDGFNet."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.sdgfnet.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    decomp_level: int = Field(default=3, ge=1)
    conv_channel: int = Field(default=32, gt=0)
    gcn_depth: int = Field(default=2, ge=0)
    propalpha: float = Field(default=0.3, ge=0.0, le=1.0)
    static_graph: Literal["rbf", "pcc"] = "rbf"
    rbf_sigma: float = Field(default=1.0, gt=0.0)
    fusion_dim: int = Field(default=128, gt=0)
    inception_width: int = Field(default=512, gt=0)


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="SDGFNet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_sdgfnet.toml",
    capabilities=frozenset({"time-series"}),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
