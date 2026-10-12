"""Model specification for GTS."""

from __future__ import annotations

from tsflab.experiments.runner.objective import TrainingBatch

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gts.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Validated GTS parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset and
    need not be declared in the TOML. ``enc_in`` (= number of nodes ``N``) is the
    only required field and serves as the fallback for ``num_nodes``.
    """

    enc_in: int = Field(ge=1)
    input_dim: int = Field(default=3, ge=1)
    rnn_units: int = Field(default=16, ge=1)
    num_rnn_layers: int = Field(default=1, ge=1)
    max_diffusion_step: int = Field(default=2, ge=1)
    embedding_dim: int = Field(default=16, ge=1)
    temp: float = Field(default=0.5, gt=0)
    prior_strength: float = Field(default=0.1, ge=0)


def build_model(cfg, params):
    """Construct GTS from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params["input_dim"],
        rnn_units=params["rnn_units"],
        num_rnn_layers=params["num_rnn_layers"],
        max_diffusion_step=params["max_diffusion_step"],
        embedding_dim=params["embedding_dim"],
        temp=params["temp"],
        prior_strength=params["prior_strength"],
    )


def training_objective(model, batch: TrainingBatch, criterion):
    """Forecast criterion plus the graph prior BCE when a prior adjacency exists."""
    forecast = batch.forecast(model)
    loss = criterion(batch.align(forecast), batch.target)
    if model.has_prior:
        loss = loss + model.graph_prior_loss(batch.x)
    return forecast, loss


SPEC = ModelSpec(
    name="GTS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("channel_alignment", "graph_conv_gru", "marks"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
    training_objective=training_objective,
)
