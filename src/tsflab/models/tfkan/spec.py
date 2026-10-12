"""Runtime specification for TFKAN."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.tfkan.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    embed_size: int = Field(default=128, gt=0)
    hidden_size: int = Field(default=256, gt=0)
    grid_size: int = Field(default=2, gt=0)
    spline_order: int = Field(default=1, ge=1)
    scale_noise: float = Field(default=0.1, ge=0.0)
    sparsity_threshold: float = Field(default=0.001, ge=0.0)
    use_bias: bool = True


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="TFKAN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_tfkan.toml",
    capabilities=frozenset({"time-series"}),
    components=("bspline_basis",),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
