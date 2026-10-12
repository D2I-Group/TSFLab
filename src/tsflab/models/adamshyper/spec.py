"""Runtime specification for AdaMSHyper."""

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.adamshyper.model import Model

"""Validated parameters for AdaMSHyper."""

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    window_size: list[int] = [4, 4]
    hyper_num: list[int] = [50, 20, 10]
    d_embed: int = 16
    eta: int = 3
    beta: float = 0.5
    gamma: float = 4.2
    lambda_balance: float = 0.5
    const_weight: float = 1.0
    dropout: float = 0.1


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
