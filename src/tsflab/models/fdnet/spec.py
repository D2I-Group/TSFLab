"""Runtime specification for FDNet."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.fdnet.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    d_model: int = Field(default=8, ge=1)
    kernel_size: int = Field(default=3, ge=1)
    num_layers: int = Field(default=5, ge=1)
    focal_parts: int = Field(default=5, ge=1)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        **{name: params[name] for name in _FIELDS if name in params},
    )


SPEC = ModelSpec(
    name="FDNet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_fdnet.toml",
    capabilities=frozenset({"time-series"}),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
