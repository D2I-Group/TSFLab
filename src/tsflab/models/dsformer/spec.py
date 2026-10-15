"""Runtime specification for DSFormer."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dsformer.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    num_layer: int = Field(1, gt=0)
    muti_head: int = Field(2, gt=0)
    num_samp: int = Field(2, gt=0)
    dropout: float = Field(0.15, ge=0, lt=1)
    if_node: bool = True


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        label_len=cfg.task.label_len,
        features=cfg.task.features,
        **params,
    )


SPEC = ModelSpec(
    name="DSFormer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
