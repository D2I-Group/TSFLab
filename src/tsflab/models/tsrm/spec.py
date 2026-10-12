"""Runtime specification for TSRM."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.tsrm.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    d_model: int = Field(default=64, ge=1)
    n_heads: int = Field(default=16, ge=1)
    e_layers: int = Field(default=7, ge=1)
    # Representation-layer convolutions as [kernel, dilation, groups]; groups -1 = depthwise.
    conv_layers: list[tuple[int, int, int]] = Field(
        default_factory=lambda: [(3, 1, -1), (5, 2, -1), (10, 3, -1)], min_length=1
    )
    attention: Literal["softmax", "entmax15"] = "entmax15"
    dropout: float = Field(default=0.25, ge=0.0, lt=1.0)
    inter_feature: bool = False

    @model_validator(mode="after")
    def _heads(self) -> "ModelParameterConfig":
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        return self

    @model_validator(mode="after")
    def _convolutions(self) -> "ModelParameterConfig":
        for kernel, dilation, groups in self.conv_layers:
            if kernel < 1 or dilation < 1:
                raise ValueError("every convolution must fit the input window")
            if not _valid_groups(groups, self.d_model):
                raise ValueError("convolution groups must be -1 (depthwise) or divide d_model")
        return self


def _valid_groups(groups: int, d_model: int) -> bool:
    return groups == -1 or (groups >= 1 and d_model % groups == 0)


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    values = {name: params[name] for name in _FIELDS if name in params}
    if "conv_layers" in values:
        values["conv_layers"] = tuple(tuple(layer) for layer in values["conv_layers"])
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **values)


SPEC = ModelSpec(
    name="TSRM",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_tsrm.toml",
    capabilities=frozenset({"time-series"}),
    components=("flatten_forecast_head", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
