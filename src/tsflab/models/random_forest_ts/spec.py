"""Model specification for RandomForestTS."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.random_forest_ts.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    num_estimators: int = Field(default=16, gt=0)
    tree_depth: int = Field(default=3, gt=0)
    feature_fraction: float = Field(default=0.7, gt=0, le=1)
    temperature: float = Field(default=1.0, gt=0)
    random_seed: int = 1729
    use_revin: bool = True


def build_model(cfg, params):
    """Construct RandomForestTS from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        num_estimators=params["num_estimators"],
        tree_depth=params["tree_depth"],
        feature_fraction=params["feature_fraction"],
        temperature=params["temperature"],
        random_seed=params["random_seed"],
        use_revin=bool(params["use_revin"]),
    )


SPEC = ModelSpec(
    name="RandomForestTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin", "soft_tree"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
