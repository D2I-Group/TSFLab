"""Model specification for FTP."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.ftp.model import Model
from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    num_layers: int = Field(default=2, gt=0)
    patch_unit: int = Field(default=4, gt=0)
    num_scales: int = Field(default=3, gt=0)
    stride: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    use_revin: bool = True


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        params["d_model"],
        params["num_layers"],
        params["patch_unit"],
        params["num_scales"],
        params["stride"],
        params["dropout"],
        bool(params["use_revin"]),
    )


SPEC = ModelSpec(
    name="FTP",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("revin",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
