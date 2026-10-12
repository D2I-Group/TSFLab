"""Runtime specification for FreTS."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.frets.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    embed_size: int = Field(default=128, gt=0)
    hidden_size: int = Field(default=256, gt=0)
    channel_independence: bool = False
    sparsity_threshold: float = Field(default=0.01, ge=0.0)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        features=cfg.task.features,
        **params,
    )


SPEC = ModelSpec(
    name="FreTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
