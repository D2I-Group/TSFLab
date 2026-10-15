"""Model specification for HN_MVTS."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.hn_mvts.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    embedding_dim: int = Field(default=8, gt=0)
    hyper_hidden: int = Field(default=32, gt=0)
    use_revin: bool = True


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params["d_model"],
        params["embedding_dim"],
        params["hyper_hidden"],
        bool(params["use_revin"]),
    )


SPEC = ModelSpec(
    name="HN_MVTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
