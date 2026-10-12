"""Runtime specification for FDNet."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

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

    @model_validator(mode="after")
    def _layers_and_kernel(self) -> "ModelParameterConfig":
        if self.num_layers < max(self.focal_parts - 1, 1):
            raise ValueError("num_layers must be at least max(focal_parts - 1, 1)")
        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        return self


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
