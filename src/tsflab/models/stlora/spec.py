"""Runtime specification for STLoRA."""

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.stlora.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``enc_in`` is the number of nodes. ``adj_mx``/``num_nodes`` injected by the
    runner are not used: the LSTM backbone and the adapters are graph-free.
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(ge=1)
    cov_dim: int = Field(default=2, ge=0)
    num_blocks: int = Field(default=1, ge=1)
    num_layers: int = Field(default=4, ge=1)
    rank: int = Field(default=16, ge=1)
    lora_alpha: float = Field(default=16.0, gt=0)
    nsp_hidden: int = Field(default=64, ge=1)
    nsp_dropout: float = Field(default=0.3, ge=0.0, lt=1.0)
    init_dim: int = Field(default=32, ge=1)
    hid_dim: int = Field(default=64, ge=1)
    end_dim: int = Field(default=512, ge=1)
    backbone_layers: int = Field(default=2, ge=1)
    backbone_dropout: float = Field(default=0.3, ge=0.0, lt=1.0)


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        **{name: params[name] for name in _FIELDS if name in params},
    )


SPEC = ModelSpec(
    name="STLoRA",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_stlora.toml",
    capabilities=frozenset({"spatiotemporal"}),
    components=("marks",),
    contract_task={"seq_len": 12, "pred_len": 12, "label_len": 0},
)
