"""Model specification for STPGNN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stpgnn.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Validated STPGNN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset and
    must not be declared in the TOML. ``enc_in`` (= number of nodes ``N``) is the
    required spatial dimension; the remaining fields are kept small so the smoke
    run stays fast.
    """

    enc_in: int
    dropout: float = 0.1
    topk: int = 4
    residual_channels: int = Field(default=16, gt=0)
    end_channels: int = 64
    kernel_size: int = 2
    blocks: int = Field(default=2, gt=0)
    layers: int = Field(default=2, gt=0)
    dims: int = 16


def build_model(cfg, params):
    """Construct STPGNN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        dropout=params["dropout"],
        topk=params["topk"],
        residual_channels=params["residual_channels"],
        end_channels=params["end_channels"],
        kernel_size=params["kernel_size"],
        blocks=params["blocks"],
        layers=params["layers"],
        dims=params["dims"],
    )


SPEC = ModelSpec(
    name="STPGNN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
