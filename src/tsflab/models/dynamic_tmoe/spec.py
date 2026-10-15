"""Model specification for DynamicTMoE."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.dynamic_tmoe.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    top_k: int = Field(default=3, gt=0, le=5)
    memory_slots: int = Field(default=4, gt=0)
    relation_period: int = Field(default=24, gt=0)
    routing_floor: float = Field(default=1e-4, ge=0, lt=1)
    use_revin: bool = True


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params["d_model"],
        params["patch_len"],
        params["stride"],
        params["top_k"],
        params["memory_slots"],
        params["relation_period"],
        params["routing_floor"],
        bool(params["use_revin"]),
    )


SPEC = ModelSpec(
    name="DynamicTMoE",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("marks", "revin", "topk_expert_router"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
