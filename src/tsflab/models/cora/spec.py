"""Model specification for CoRA."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.cora.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    rank: int = Field(default=4, gt=0)
    polynomial_order: int = Field(default=2, ge=0)
    use_revin: bool = True


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params["d_model"],
        params["rank"],
        params["polynomial_order"],
        bool(params["use_revin"]),
    )


SPEC = ModelSpec(
    name="CoRA",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
