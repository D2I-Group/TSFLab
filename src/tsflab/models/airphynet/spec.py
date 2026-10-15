"""Runtime specification for AirPhyNet."""

from typing import Literal

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.airphynet.model import Model


class ModelParameterConfig(BaseModel):
    enc_in: int = Field(gt=0)
    cov_dim: int = Field(default=2, gt=0)
    latent_dim: int = Field(default=8, gt=0)
    rnn_units: int = Field(default=32, gt=0)
    ode_method: Literal["euler", "rk4"] = "rk4"


def build_model(cfg, params):
    params.pop("num_nodes", None)
    return Model(
        cfg.task.seq_len,
        cfg.task.pred_len,
        adj_mx=params.pop("adj_mx", None),
        flow_mx=params.pop("flow_mx", None),
        **params,
    )


SPEC = ModelSpec(
    name="AirPhyNet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["covariate"]),
    components=("marks",),
    contract_task={"seq_len": 24, "pred_len": 24, "label_len": 0},
    requires_graph=True,
)
