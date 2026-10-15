"""Runtime specification for CrossGNN."""

from pydantic import BaseModel, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.crossgnn.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    e_layers: int = Field(default=2, gt=0)
    anti_ood: bool = True
    tk: int = Field(default=3, ge=2)
    scale_number: int = Field(default=4, gt=0)
    use_tgcn: bool = True
    use_ngcn: bool = True
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    tvechidden: int = Field(default=8, gt=0)
    nvechidden: int = Field(default=8, gt=0)
    hidden: int = Field(default=16, gt=0)

    @model_validator(mode="after")
    def _graph_contract(self) -> "ModelParameterConfig":
        if not self.use_tgcn and not self.use_ngcn:
            raise ValueError("at least one graph path must be enabled")
        if self.enc_in < 2 * self.tk:
            raise ValueError("enc_in must be at least 2 * tk")
        return self


def build_model(cfg, params):
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **params)


SPEC = ModelSpec(
    name="CrossGNN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["time-series"]),
    components=("last_value_center",),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
    contract_seeds=(0, 18, 24),
)
