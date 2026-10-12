"""Runtime specification for ReNF."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.renf.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    variant: Literal["alpha", "beta"] = "beta"
    num_blocks: int = Field(default=3, ge=1)
    d_ff: int = Field(default=2048, ge=1)
    dropout: float = Field(default=0.5, ge=0.0, lt=1.0)
    body_layers: int = Field(default=1, ge=1)
    block_norm: Literal["layer", "instance"] = "layer"
    use_pe: bool = True
    alpha_freq: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _variant_contract(self) -> "ModelParameterConfig":
        if self.variant == "alpha" and (self.block_norm != "layer" or self.body_layers != 1):
            raise ValueError("ReNF-alpha uses LayerNorm blocks with one transform layer")
        return self


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        **{name: params[name] for name in _FIELDS if name in params},
    )


def training_objective(model, batch, criterion):
    """Eq. (4): block-wise time/frequency L1 supervision of every sub-forecast.

    Each sub-forecast covers the first ``k * H / N`` target steps; the final
    full-horizon sub-forecast is the returned point forecast.
    """
    forecasts = model.sub_forecasts(batch.x)
    channels = slice(-1, None) if batch.features == "MS" else slice(0, None)
    target = batch.target
    loss = model.training_loss([forecast[..., channels] for forecast in forecasts], target)
    return forecasts[-1], loss


SPEC = ModelSpec(
    name="ReNF",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_renf.toml",
    capabilities=frozenset({"time-series"}),
    components=("positional_encoding", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    training_objective=training_objective,
)
