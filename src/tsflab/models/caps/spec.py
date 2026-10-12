"""Runtime specification for CAPS."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.caps.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``."""

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    d_model: int = Field(default=16, ge=1)
    value_dim: int = Field(default=16, ge=1)
    n_heads: int = Field(default=4, ge=1)
    e_layers: int = Field(default=3, ge=1)
    dropout: float = Field(default=0.0, ge=0.0, lt=1.0)
    channel_dropout: bool = True
    learn_rope_freq: bool = True

    @model_validator(mode="after")
    def _even_heads(self) -> "ModelParameterConfig":
        width = self.d_model + self.value_dim
        if width % self.n_heads or (width // self.n_heads) % 2:
            raise ValueError("d_model + value_dim must split into n_heads heads of even width")
        return self


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    values = {name: params[name] for name in _FIELDS if name in params}
    return Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, **values)


SPEC = ModelSpec(
    name="CAPS",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_caps.toml",
    capabilities=frozenset({"time-series"}),
    components=("last_value_center", "mamba"),
    contract_task={"seq_len": 96, "pred_len": 96, "label_len": 0},
)
