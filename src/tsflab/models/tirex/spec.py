"""Runtime specification for TiRex."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.tirex.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    patch_len: int = Field(default=16, gt=0)
    num_layers: int = Field(default=2, gt=0)
    dropout: float = 0.1
    quantile_levels: list[float] | None = None


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        features=cfg.task.features,
        d_model=params["d_model"],
        patch_len=params["patch_len"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        quantile_levels=params.get("quantile_levels") or list(cfg.evaluation.quantile_levels),
    )


SPEC = ModelSpec(
    name="TiRex",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["quantile-output", "time-series"]),
    components=("instance_norm", "quantile_head"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
