"""Runtime specification for AdaMSHyper."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.adamshyper.model import Model

"""Validated parameters for AdaMSHyper."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    window_size: list[int] = [4, 4]
    hyper_num: list[int] = [50, 20, 10]
    d_embed: int = Field(default=16, gt=0)
    eta: int = Field(default=3, gt=0)
    beta: float = Field(default=0.5, ge=0.0, lt=1.0)
    gamma: float = Field(default=4.2, gt=0)
    lambda_balance: float = Field(default=0.5, ge=0.0, le=1.0)
    const_weight: float = Field(default=1.0, ge=0)
    dropout: float = 0.1

    @model_validator(mode="after")
    def _scales(self) -> "ModelParameterConfig":
        if len(self.hyper_num) != len(self.window_size) + 1 or min(self.hyper_num + self.window_size, default=1) < 1:
            raise ValueError("hyper_num needs one entry per scale (len(window_size) + 1)")
        return self


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        enc_in=params["enc_in"],
        window_size=params["window_size"],
        hyper_num=params["hyper_num"],
        d_embed=params["d_embed"],
        eta=params["eta"],
        beta=params["beta"],
        gamma=params["gamma"],
        lambda_balance=params["lambda_balance"],
        const_weight=params["const_weight"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="AdaMSHyper",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_adamshyper.toml",
    capabilities=frozenset({"time-series"}),
    components=("adaptive_node_embedding_adjacency", "revin"),
    contract_task={"seq_len": 96, "pred_len": 12, "label_len": 0},
)
