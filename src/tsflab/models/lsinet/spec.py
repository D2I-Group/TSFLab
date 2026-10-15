"""Runtime specification for LSINet."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.lsinet.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    c_out: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    n_heads: int = Field(default=4, gt=0)
    e_layers: int = Field(default=1, gt=0)
    n_patches: int = Field(default=64, gt=0)
    density: float = Field(default=0.15, gt=0.0, le=1.0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    use_norm: bool = True

    @model_validator(mode="after")
    def _architecture_contract(self) -> "ModelParameterConfig":
        if self.enc_in != self.c_out:
            raise ValueError("LSINet is channel-independent: enc_in and c_out must match")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        c_out=params["c_out"],
        d_model=params["d_model"],
        n_heads=params["n_heads"],
        e_layers=params["e_layers"],
        n_patches=params["n_patches"],
        density=params["density"],
        dropout=params["dropout"],
        use_norm=params["use_norm"],
    )


SPEC = ModelSpec(
    name="LSINet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_lsinet.toml",
    capabilities=frozenset({"time-series"}),
    components=(
        "flatten_forecast_head",
        "positional_encoding",
        "revin",
        "sparse_connection_router",
    ),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
