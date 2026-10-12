"""Model specification for TimeMosaic."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.timemosaic.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    patch_sizes: tuple[int, ...] = (4, 8, 16)
    num_segments: int = Field(default=4, gt=0)
    num_heads: int = Field(default=4, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    use_revin: bool = True
    selection_tau: float = Field(default=0.5, gt=0)
    budget_weight: float = Field(default=0.001, ge=0)


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params["d_model"],
        tuple(params["patch_sizes"]),
        params["num_segments"],
        params["num_heads"],
        params["dropout"],
        bool(params["use_revin"]),
        float(params["selection_tau"]),
        float(params["budget_weight"]),
    )


SPEC = ModelSpec(
    name="TimeMosaic",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
