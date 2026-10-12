"""Model specification for Extralonger."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.extralonger.model import Model

from pydantic import BaseModel, ConfigDict


class ModelParameterConfig(BaseModel):
    """Validated Extralonger parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``src/tsflab/experiments/runner/run_one.py``) and need not be declared in the
    TOML. ``enc_in`` (the node count ``N``) is the only required field and is
    used as the fallback node count. Defaults are modest for fast CPU smoke
    runs; the paper's PEMS04/PEMS08/SEATTLE presets use much wider embeddings
    (see the model card).
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    input_dim: int = 3
    steps_per_day: int = 24
    input_embedding_dim: int = 8
    tod_embedding_dim: int = 4
    dow_embedding_dim: int = 4
    spatial_embedding_dim: int = 8
    feed_forward_dim: int = 16
    num_heads: int = 2
    num_layers: int = 1
    dropout: float = 0.0


def build_model(cfg, params):
    """Construct Extralonger from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params["input_dim"],
        steps_per_day=params["steps_per_day"],
        input_embedding_dim=params["input_embedding_dim"],
        tod_embedding_dim=params["tod_embedding_dim"],
        dow_embedding_dim=params["dow_embedding_dim"],
        spatial_embedding_dim=params["spatial_embedding_dim"],
        feed_forward_dim=params["feed_forward_dim"],
        num_heads=params["num_heads"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
    )


SPEC = ModelSpec(
    name="Extralonger",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("graph_masked_attention", "marks"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
