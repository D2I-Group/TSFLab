"""Runtime specification for AdaMamba."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.adamamba.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    d_model: int = Field(default=32, ge=4)
    d_freq: int = Field(default=32, ge=1)
    e_layers: int = Field(default=2, ge=1)
    patch_lens: list[int] = Field(default_factory=lambda: [96, 18, 9], min_length=1)
    stride: int = Field(default=16, ge=1)
    conv_kernel: int = Field(default=3, ge=1)
    beta: float = 1.0
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    head_dropout: float = Field(default=0.0, ge=0.0, lt=1.0)


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    values = {name: params[name] for name in _FIELDS if name in params}
    if "patch_lens" in values:
        values["patch_lens"] = tuple(values["patch_lens"])
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **values)


SPEC = ModelSpec(
    name="AdaMamba",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_adamamba.toml",
    capabilities=frozenset({"time-series"}),
    components=("flatten_forecast_head", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
