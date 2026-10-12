"""Runtime specification for TSRAG."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.tsrag.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    top_k: int = 4
    memory_size: int = 8
    num_heads: int = 4
    dropout: float = 0.1


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        d_model=params["d_model"],
        top_k=params["top_k"],
        memory_size=params["memory_size"],
        num_heads=params["num_heads"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="TSRAG",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
