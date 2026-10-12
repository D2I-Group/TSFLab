"""Model specification for ExtraTreesTS."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.extra_trees_ts.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    num_estimators: int = Field(default=24, gt=0)
    tree_depth: int = Field(default=2, gt=0)
    threshold_range: float = Field(default=1.0, gt=0)
    temperature: float = Field(default=1.0, gt=0)
    random_seed: int = 1733
    use_revin: bool = True


def build_model(cfg, params):
    """Construct ExtraTreesTS from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        num_estimators=params["num_estimators"],
        tree_depth=params["tree_depth"],
        threshold_range=params["threshold_range"],
        temperature=params["temperature"],
        random_seed=params["random_seed"],
        use_revin=bool(params["use_revin"]),
    )


SPEC = ModelSpec(
    name="ExtraTreesTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin", "soft_tree"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
