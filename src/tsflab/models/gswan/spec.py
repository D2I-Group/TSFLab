"""Runtime specification for GSWaN."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.gswan.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``num_nodes`` and the predefined road graph ``adj_mx`` are injected by the
    runner from the dataset and are therefore not declared here.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    hidden_channels: int = Field(default=40, gt=0)
    skip_channels: int = Field(default=320, gt=0)
    end_channels: int = Field(default=640, gt=0)
    kernel_size: int = Field(default=2, ge=2)
    blocks: int = Field(default=4, gt=0)
    layers: int = Field(default=2, gt=0)
    num_heads: int = Field(default=3, ge=0)
    order: int = Field(default=2, gt=0)
    softmax_temp: float = Field(default=7.0, gt=0.0)
    dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    time_feature: Literal["time_of_week", "time_of_day"] = "time_of_week"


_FIELDS = tuple(name for name in ModelParameterConfig.model_fields if name != "enc_in")


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        num_nodes=params.get("num_nodes", params["enc_in"]),
        adj_mx=params.get("adj_mx"),
        **{name: params[name] for name in _FIELDS if name in params},
    )


SPEC = ModelSpec(
    name="GSWaN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_gswan.toml",
    capabilities=frozenset({"spatiotemporal"}),
    components=("adaptive_node_embedding_adjacency", "graph_utils", "marks"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
