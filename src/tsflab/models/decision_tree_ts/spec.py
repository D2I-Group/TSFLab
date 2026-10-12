"""Model specification for DecisionTreeTS."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.decision_tree_ts.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    tree_depth: int = Field(default=4, gt=0)
    temperature: float = Field(default=1.0, gt=0)
    use_revin: bool = True


def build_model(cfg, params):
    """Construct DecisionTreeTS from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        tree_depth=params["tree_depth"],
        temperature=params["temperature"],
        use_revin=bool(params["use_revin"]),
    )


SPEC = ModelSpec(
    name="DecisionTreeTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin", "soft_tree"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
