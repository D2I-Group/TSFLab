"""Runtime specification for TwinFormer."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.twinformer.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    d_model: int = Field(default=128, ge=1)
    n_heads: int = Field(default=4, ge=1)
    d_ff: int = Field(default=512, ge=1)
    patch_len: int = Field(default=12, ge=1)
    top_k: int = Field(default=5, ge=1)
    local_layers: int = Field(default=1, ge=1)
    global_layers: int = Field(default=1, ge=1)
    dropout: float = Field(default=0.15, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _heads_divide_width(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    values = {name: params[name] for name in _FIELDS if name in params}
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **values)


SPEC = ModelSpec(
    name="TwinFormer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_twinformer.toml",
    capabilities=frozenset({"time-series"}),
    components=("embed",),
    contract_task={"seq_len": 48, "pred_len": 96, "label_len": 0},
)
