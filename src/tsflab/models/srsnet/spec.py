"""Runtime specification for SRSNet."""

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.srsnet.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    patch_len: int = Field(default=24, gt=0)
    stride: int = Field(default=24, gt=0)
    hidden_size: int = Field(default=64, gt=0)
    dropout: float = Field(default=0.2, ge=0, lt=1)
    head_dropout: float = Field(default=0.1, ge=0, lt=1)
    alpha: float = Field(default=2.0, gt=0)
    pos: bool = True
    head_mode: str = Field(default="linear", pattern="^linear$")
    affine: bool = True
    subtract_last: bool = False


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        features=cfg.task.features,
        d_model=params["d_model"],
        patch_len=params["patch_len"],
        stride=params["stride"],
        hidden_size=params["hidden_size"],
        dropout=params["dropout"],
        head_dropout=params["head_dropout"],
        alpha=params["alpha"],
        pos=params["pos"],
        head_mode=params["head_mode"],
        affine=params["affine"],
        subtract_last=params["subtract_last"],
    )


SPEC = ModelSpec(
    name="SRSNet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("flatten_forecast_head", "revin"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
