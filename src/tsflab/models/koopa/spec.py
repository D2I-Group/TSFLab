"""Runtime specification for Koopa."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.koopa.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    seg_len: int | None = Field(default=None, gt=1)
    dynamic_dim: int = Field(default=128, gt=0)
    hidden_dim: int = Field(default=64, gt=0)
    hidden_layers: int = Field(default=2, gt=0)
    num_blocks: int = Field(default=3, gt=0)
    multistep: bool = False
    alpha: float = Field(default=0.2, gt=0, le=1)


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        seg_len=params.get("seg_len"),
        dynamic_dim=params["dynamic_dim"],
        hidden_dim=params["hidden_dim"],
        hidden_layers=params["hidden_layers"],
        num_blocks=params["num_blocks"],
        multistep=params["multistep"],
        alpha=params["alpha"],
    )


SPEC = ModelSpec(
    name="Koopa",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 192, "pred_len": 96, "label_len": 0},
)
