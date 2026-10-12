"""Model specification for STWave."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stwave.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STWave parameters supplied via ``model.params``.

    ``enc_in`` (the number of spatial nodes ``N``) is required. ``num_nodes``
    and ``adj_mx`` are injected by the runner from the dataset and are NOT
    declared in the TOML. ``hidden_size`` doubles as the number of Laplacian
    eigenvectors used for the spatial positional encoding, so it is clamped to
    ``N`` at construction.
    """

    enc_in: int
    hidden_size: int = 16
    layers: int = 1
    log_samples: int = 1


def build_model(cfg, params):
    """Construct STWave from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        hidden_size=params["hidden_size"],
        layers=params["layers"],
        log_samples=params["log_samples"],
    )


SPEC = ModelSpec(
    name="STWave",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("marks",),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
