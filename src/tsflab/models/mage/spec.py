"""Model specification for MAGE."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.mage.model import Model

from pydantic import BaseModel, Field, model_validator


class ModelParameterConfig(BaseModel):
    """Validated MAGE parameters supplied via ``model.params``."""

    enc_in: int = Field(gt=0)
    model_dim: int = Field(default=64, gt=0)
    recur_num: int = Field(default=8, gt=0)
    topk: int = Field(default=2, gt=0)
    node_dim: int = Field(default=16, gt=0)

    @model_validator(mode="after")
    def _topk_within_experts(self):
        if self.topk > self.recur_num:
            raise ValueError("MAGE topk must be in [1, recur_num]")
        return self


def build_model(cfg, params):
    """Construct MAGE from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        model_dim=params["model_dim"],
        recur_num=params["recur_num"],
        topk=params["topk"],
        node_dim=params["node_dim"],
    )


SPEC = ModelSpec(
    name="MAGE",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("marks",),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
