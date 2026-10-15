"""Model specification for VisiFold."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.models.visifold.model import Model


class ModelParameterConfig(BaseModel):
    """Validated VisiFold parameters supplied via ``model.params``.

    ``enc_in`` (= number of nodes ``N``) is required. ``num_nodes`` and
    ``adj_mx`` are injected by the runner from the dataset and are not
    declared here. Defaults are modest for fast smoke runs.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int
    input_dim: int = Field(default=3, gt=0)
    steps_per_day: int = 24
    input_embedding_dim: int = Field(default=8, gt=0)
    tod_embedding_dim: int = 4
    dow_embedding_dim: int = 4
    spatial_embedding_dim: int = 8
    feed_forward_dim: int = 16
    num_heads: int = Field(default=2, gt=0)
    num_layers: int = Field(default=1, gt=0)
    mask_ratio: float = Field(default=0.2, ge=0.0, lt=1.0)
    subgraph_size: int = Field(default=4, gt=0)
    dropout: float = 0.0

    @model_validator(mode="after")
    def _check_heads(self):
        model_dim = (
            self.input_embedding_dim + self.tod_embedding_dim + self.dow_embedding_dim + self.spatial_embedding_dim
        )
        if model_dim % self.num_heads:
            raise ValueError("VisiFold model_dim must be divisible by num_heads")
        return self


def build_model(cfg, params):
    """Construct VisiFold from a validated run configuration."""
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
        mask_ratio=params["mask_ratio"],
        subgraph_size=params["subgraph_size"],
        dropout=params["dropout"],
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Forecast criterion restricted to the nodes kept by node visibility."""
    forecast = batch.forecast(model)
    outputs, target = batch.align(forecast), batch.target
    keep = model.last_keep_indices
    if keep is None:
        return forecast, criterion(outputs, target)
    offset = model.num_nodes - target.shape[-1]
    keep = keep[keep >= offset] - offset
    if keep.numel() == 0:
        return forecast, outputs.sum() * 0.0
    return forecast, criterion(outputs[..., keep], target[..., keep])


SPEC = ModelSpec(
    name="VisiFold",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("marks", "node_visibility"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
)
