"""Runtime specification for PRISM."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.prism.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    num_components: int = Field(default=5, ge=2)
    tree_depth: int = Field(default=2, ge=1, le=6)
    overlap: int = Field(default=8, ge=0)
    use_last_layer_only: bool = False


def build_model(cfg, params):
    options = ModelParameterConfig(**params).model_dump()
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **options)


SPEC = ModelSpec(
    name="PRISM",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_prism.toml",
    capabilities=frozenset({"time-series"}),
    contract_task={"seq_len": 336, "pred_len": 96, "label_len": 0},
)
