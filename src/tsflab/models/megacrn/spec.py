"""Model specification for MegaCRN."""

from __future__ import annotations

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.megacrn.model import Model

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Validated MegaCRN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are *injected* by the runner from the dataset
    (see ``src/tsflab/experiments/runner/run_one.py``) and need not be declared in TOML.
    """

    enc_in: int  # number of spatial nodes N (required)
    input_dim: int = Field(default=3, gt=0)
    rnn_units: int = Field(default=32, gt=0)
    num_layers: int = Field(default=1, gt=0)
    cheb_k: int = Field(default=3, gt=0)
    mem_num: int = Field(default=8, gt=0)
    mem_dim: int = Field(default=16, gt=0)


def build_model(cfg, params):
    """Construct MegaCRN from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        input_dim=params["input_dim"],
        rnn_units=params["rnn_units"],
        num_layers=params["num_layers"],
        cheb_k=params["cheb_k"],
        mem_num=params["mem_num"],
        mem_dim=params["mem_dim"],
    )


SPEC = ModelSpec(
    name="MegaCRN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset(["spatiotemporal"]),
    components=("graph_conv_gru", "marks"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
