"""Model specification for STID."""

from __future__ import annotations

from pydantic import BaseModel, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stid.model import Model


class ModelParameterConfig(BaseModel):
    """Validated STID parameters supplied via ``model.params``.

    ``enc_in`` (= number of nodes ``N``) is required. ``num_nodes`` and
    ``adj_mx`` are injected by the runner from the dataset, so they are not
    declared here. Defaults are modest for fast smoke runs.
    """

    enc_in: int
    input_dim: int = Field(default=3, gt=0)
    embed_dim: int = Field(default=32, gt=0)
    num_layers: int = 1
    num_time_in_day: int = 24
    num_day_in_week: int = 7
    if_time_in_day: bool = True
    if_day_in_week: bool = True


def build_model(cfg, params):
    """Construct STID from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params["input_dim"],
        embed_dim=params["embed_dim"],
        num_layers=params["num_layers"],
        num_time_in_day=params["num_time_in_day"],
        num_day_in_week=params["num_day_in_week"],
        if_time_in_day=params["if_time_in_day"],
        if_day_in_week=params["if_day_in_week"],
    )


SPEC = ModelSpec(
    name="STID",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("marks",),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
