"""Model specification for COSA."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.cosa.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    context_len: int = Field(default=10, gt=0)
    gate_init: float = 0.1


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params["context_len"],
        params["gate_init"],
    )


SPEC = ModelSpec(
    name="COSA",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series", "test-time-adaptation"]),
    components=("channel_wise_linear",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
