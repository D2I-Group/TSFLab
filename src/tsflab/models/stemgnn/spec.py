"""Model specification for StemGNN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stemgnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated StemGNN parameters supplied via ``model.params``.

    ``enc_in`` is the number of nodes. StemGNN learns its graph from history and
    therefore does not consume the dataset adjacency matrix.
    """

    enc_in: int
    multi_layer: int = 3
    dropout_rate: float = 0.5
    leaky_rate: float = 0.2


def build_model(cfg, params):
    """Construct StemGNN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        multi_layer=params["multi_layer"],
        dropout_rate=params["dropout_rate"],
        leaky_rate=params["leaky_rate"],
    )


SPEC = ModelSpec(
    name="StemGNN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
