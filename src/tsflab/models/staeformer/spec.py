"""Model specification for STAEformer."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.staeformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STAEformer parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``src/tsflab/experiments/runner/run_one.py``) and need not be declared in the
    TOML. ``enc_in`` (the node count ``N``) is the only required field and is
    used as the fallback node count.
    """

    enc_in: int
    input_dim: int = 3
    steps_per_day: int = 24
    input_embedding_dim: int = 24
    tod_embedding_dim: int = 24
    dow_embedding_dim: int = 24
    spatial_embedding_dim: int = 0
    adaptive_embedding_dim: int = 80
    feed_forward_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    use_mixed_proj: bool = True


def build_model(cfg, params):
    """Construct STAEformer from a validated run configuration."""
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
        adaptive_embedding_dim=params["adaptive_embedding_dim"],
        feed_forward_dim=params["feed_forward_dim"],
        num_heads=params["num_heads"],
        num_layers=params["num_layers"],
        dropout=params["dropout"],
        use_mixed_proj=params["use_mixed_proj"],
    )


SPEC = ModelSpec(
    name="STAEformer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("marks",),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
