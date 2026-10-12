"""Runtime specification for FAITH."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.faith.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    d_model: int = Field(default=32, ge=1)
    n_heads: int = Field(default=4, ge=1)
    e_layers: int = Field(default=1, ge=1)
    d_ff: int = Field(default=256, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    head_dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    modes: int = Field(default=64, ge=1)
    kernel_sizes: list[int] = Field(default_factory=lambda: [17, 49], min_length=1)

    @model_validator(mode="after")
    def _heads_divide_width(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    values = {name: params[name] for name in _FIELDS if name in params}
    if "kernel_sizes" in values:
        values["kernel_sizes"] = tuple(values["kernel_sizes"])
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **values)


SPEC = ModelSpec(
    name="FAITH",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_faith.toml",
    capabilities=frozenset({"time-series"}),
    components=("flatten_forecast_head", "revin", "self_attention_family", "series_decomposition"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
