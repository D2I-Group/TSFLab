"""Runtime specification for SAMBA."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.samba.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    d_ff: int = Field(default=128, gt=0)
    e_layers: int = Field(default=1, ge=0)
    d_layers: int = Field(default=1, ge=0)
    d_state1: int = Field(default=16, gt=0)
    d_state2: int = Field(default=16, gt=0)
    patch_len: int = Field(default=16, gt=0)
    stride: int = Field(default=8, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    use_act: bool = False

    @model_validator(mode="after")
    def _at_least_one_encoder(self) -> "ModelParameterConfig":
        if self.e_layers + self.d_layers == 0:
            raise ValueError("at least one of e_layers and d_layers must be positive")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        d_model=params["d_model"],
        d_ff=params["d_ff"],
        e_layers=params["e_layers"],
        d_layers=params["d_layers"],
        d_state1=params["d_state1"],
        d_state2=params["d_state2"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        dropout=params["dropout"],
        use_act=bool(params["use_act"]),
    )


SPEC = ModelSpec(
    name="SAMBA",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_samba.toml",
    capabilities=frozenset({"time-series"}),
    components=("flatten_forecast_head", "mamba", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
