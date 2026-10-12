"""Runtime specification for SOFTS."""

from pydantic import BaseModel, Field
from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.softs.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    d_model: int = Field(default=128, gt=0)
    d_core: int = Field(default=64, gt=0)
    d_ff: int = Field(default=256, gt=0)
    e_layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    activation: str = Field(default="gelu", pattern="^(gelu|relu)$")
    use_norm: bool = True


def build_model(cfg, params):
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        params["enc_in"],
        features=cfg.task.features,
        label_len=cfg.task.label_len,
        d_model=params["d_model"],
        d_core=params["d_core"],
        d_ff=params["d_ff"],
        e_layers=params["e_layers"],
        dropout=params["dropout"],
        activation=params["activation"],
        use_norm=params["use_norm"],
    )


SPEC = ModelSpec(
    name="SOFTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("instance_norm",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
