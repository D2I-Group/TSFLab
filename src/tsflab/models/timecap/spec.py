"""Runtime specification for TimeCAP."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timecap.model import Model
from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 32
    patch_len: int = 16
    group_size: int = 4
    group_stride: int = 2
    num_heads: int = 4
    dropout: float = 0.1
    fusion_alpha: float = 0.1
    fusion_midpoint: float | None = None


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        d_model=params["d_model"],
        patch_len=params["patch_len"],
        group_size=params["group_size"],
        group_stride=params["group_stride"],
        num_heads=params["num_heads"],
        dropout=params["dropout"],
        fusion_alpha=params["fusion_alpha"],
        fusion_midpoint=params.get("fusion_midpoint"),
    )


SPEC = ModelSpec(
    name="TimeCAP",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
