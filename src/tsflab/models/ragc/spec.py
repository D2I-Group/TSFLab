"""Model specification for RAGC."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.ragc.model import Model


class ModelParameterConfig(BaseModel):
    """Validated RAGC parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the
    dataset, so they are not declared here. ``enc_in`` (= number of nodes
    ``N``) is required; the remaining defaults are kept modest for a fast
    spatiotemporal smoke run.
    """

    enc_in: int
    input_dim: int = Field(default=3, ge=1)
    spatial_dim: int = Field(default=16, ge=1)
    embed_dim: int = Field(default=16, ge=1)
    temp_dim_tid: int = Field(default=8, ge=1)
    temp_dim_diw: int = Field(default=8, ge=1)
    num_time_in_day: int = 24
    num_day_in_week: int = 7
    num_layer: int = Field(default=2, ge=1)
    order: int = Field(default=1, ge=1)
    sse_p: float = Field(default=0.1, ge=0.0, le=1.0)
    use_sse: bool = True


def build_model(cfg, params):
    """Construct RAGC from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params["input_dim"],
        spatial_dim=params["spatial_dim"],
        embed_dim=params["embed_dim"],
        temp_dim_tid=params["temp_dim_tid"],
        temp_dim_diw=params["temp_dim_diw"],
        num_time_in_day=params["num_time_in_day"],
        num_day_in_week=params["num_day_in_week"],
        num_layer=params["num_layer"],
        order=params["order"],
        sse_p=params["sse_p"],
        use_sse=params["use_sse"],
    )


SPEC = ModelSpec(
    name="RAGC",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("marks", "regularized_adaptive_graph_conv"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
