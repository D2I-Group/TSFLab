"""Runtime specification for GAMMANet."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gammanet.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``num_nodes`` and the predefined ``adj_mx`` are injected by the runner from the
    dataset; without an adjacency every node only attends to itself in the GAT
    layers.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    steps_per_day: int = Field(default=288, gt=0)
    input_embedding_dim: int = Field(default=24, gt=0)
    tod_embedding_dim: int = Field(default=24, ge=0)
    dow_embedding_dim: int = Field(default=24, ge=0)
    adaptive_embedding_dim: int = Field(default=80, ge=0)
    num_heads: int = Field(default=4, gt=0)
    num_layers: int = Field(default=3, gt=0)
    gat: bool = True
    d_state: int = Field(default=16, gt=0)
    d_conv: int = Field(default=4, gt=0)
    expand: int = Field(default=2, gt=0)
    scan_layout: Literal["axis", "official"] = "axis"


def build_model(cfg, params):
    options = ModelParameterConfig(
        **{key: value for key, value in params.items() if key not in ("num_nodes", "adj_mx")}
    ).model_dump()
    enc_in = options.pop("enc_in")
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", enc_in),
        adj_mx=params.get("adj_mx"),
        **options,
    )


SPEC = ModelSpec(
    name="GAMMANet",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_gammanet.toml",
    capabilities=frozenset({"spatiotemporal"}),
    components=("mamba", "marks"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
