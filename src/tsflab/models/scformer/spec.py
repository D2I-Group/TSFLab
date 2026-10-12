"""Runtime specification for SCFormer."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from tsflab.catalog.registry.models import ModelSpec
from tsflab.models.scformer.model import Model


class ModelParameterConfig(BaseModel):
    """Parameters supplied through ``model.params``.

    ``task.seq_len`` must be at least ``lookback``; the extra leading steps are the
    accumulated history summarized by the HiPPO state (official: 2047 extra steps).
    """

    model_config = ConfigDict(extra="forbid")

    enc_in: int = Field(gt=0)
    lookback: int = Field(default=96, gt=0)
    hippo_order: int = Field(default=512, gt=0)
    d_model: int = Field(default=512, gt=0)
    n_heads: int = Field(default=16, gt=0)
    e_layers: int = Field(default=2, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: Literal["gelu", "relu"] = "gelu"
    struct_mode: Literal["triangular", "conv"] = "triangular"
    kernel_size: int = Field(default=32, ge=2)
    use_norm: bool = True


_FIELDS = tuple(ModelParameterConfig.model_fields)


def build_model(cfg, params):
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        **{name: params[name] for name in _FIELDS if name in params},
    )


SPEC = ModelSpec(
    name="SCFormer",
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    smoke_config="configs/runs/smoke_scformer.toml",
    capabilities=frozenset({"time-series"}),
    components=("instance_norm", "self_attention_family"),
    contract_task={"seq_len": 192, "pred_len": 96, "label_len": 0},
)
