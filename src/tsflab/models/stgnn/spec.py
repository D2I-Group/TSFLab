"""Runtime specification for STGNN."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stgnn.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params`` (``adj_mx`` is runner-injected)."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    hidden_dim: int = Field(default=64, ge=2)
    n_heads: int = Field(default=4, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def _heads(self) -> "ModelParameterConfig":
        if self.hidden_dim % 2 or self.hidden_dim % self.n_heads:
            raise ValueError("hidden_dim must be even and divisible by n_heads")
        return self


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    values = {name: params[name] for name in _FIELDS if name in params}
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        adj_mx=params.get("adj_mx"),
        **values,
    )


SPEC = ModelSpec(
    name="STGNN",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    capabilities=frozenset({"spatiotemporal"}),
    components=("embed", "graph_conv_gru"),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
    requires_graph=True,
)
