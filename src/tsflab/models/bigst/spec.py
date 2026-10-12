"""Model specification for BigST."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.bigst.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated BigST parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    and are not declared here.
    """

    enc_in: int  # number of nodes N (required)
    input_dim: int = 3  # 1 value + calendar covariates [tod, dow]
    hid_dim: int = 16
    node_dim: int = 8
    time_dim: int = 8
    tod_size: int = 24
    dow_size: int = 7
    tau: float = 1.0
    random_feature_dim: int = 16
    dropout: float = 0.1
    use_residual: bool = True
    use_bn: bool = True


def build_model(cfg, params):
    """Construct BigST from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params["input_dim"],
        hid_dim=params["hid_dim"],
        node_dim=params["node_dim"],
        time_dim=params["time_dim"],
        tod_size=params["tod_size"],
        dow_size=params["dow_size"],
        tau=params["tau"],
        random_feature_dim=params["random_feature_dim"],
        dropout=params["dropout"],
        use_residual=params["use_residual"],
        use_bn=params["use_bn"],
    )


SPEC = ModelSpec(
    name="BigST",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("marks",),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
