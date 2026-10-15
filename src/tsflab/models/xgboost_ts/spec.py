"""Model specification for XGBoostTS."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.xgboost_ts.model import MAX_DENSE_PARAMETERS, Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    num_estimators: int = Field(default=16, gt=0)
    tree_depth: int = Field(default=3, gt=0)
    learning_rate: float = Field(default=0.1, gt=0)
    column_fraction: float = Field(default=0.8, gt=0, le=1)
    l1_penalty: float = Field(default=0.0, ge=0)
    l2_penalty: float = Field(default=0.0001, ge=0)
    temperature: float = Field(default=1.0, gt=0)
    random_seed: int = 1741
    use_revin: bool = True
    max_dense_parameters: int = Field(default=MAX_DENSE_PARAMETERS, gt=0)


def build_model(cfg, params):
    """Construct XGBoostTS from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        num_estimators=params["num_estimators"],
        tree_depth=params["tree_depth"],
        learning_rate=params["learning_rate"],
        column_fraction=params["column_fraction"],
        l1_penalty=params["l1_penalty"],
        l2_penalty=params["l2_penalty"],
        temperature=params["temperature"],
        random_seed=params["random_seed"],
        use_revin=bool(params["use_revin"]),
        max_dense_parameters=params["max_dense_parameters"],
    )


SPEC = ModelSpec(
    name="XGBoostTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin", "soft_tree"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
