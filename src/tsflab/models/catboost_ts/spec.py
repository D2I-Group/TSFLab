"""Model specification for CatBoostTS."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.catboost_ts.model import MAX_DENSE_PARAMETERS, Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    num_estimators: int = Field(default=16, gt=0)
    tree_depth: int = Field(default=3, gt=0)
    learning_rate: float = Field(default=0.1, gt=0)
    temperature: float = Field(default=1.0, gt=0)
    use_revin: bool = True
    max_dense_parameters: int = Field(default=MAX_DENSE_PARAMETERS, gt=0)


def build_model(cfg, params):
    """Construct CatBoostTS from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        num_estimators=params["num_estimators"],
        tree_depth=params["tree_depth"],
        learning_rate=params["learning_rate"],
        temperature=params["temperature"],
        use_revin=bool(params["use_revin"]),
        max_dense_parameters=params["max_dense_parameters"],
    )


SPEC = ModelSpec(
    name="CatBoostTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin", "soft_tree"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
